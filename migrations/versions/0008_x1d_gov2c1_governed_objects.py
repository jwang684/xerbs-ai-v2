"""x1d-gov2c1 governed object identity, lifecycle and evidence provenance

What was wrong
--------------
``SafetyEngine.create_relationship`` wrote ``review_status='REVIEWED'`` as a
literal, and ``create_rule`` did the same. The ORM default was DRAFT; the code
overrode it. So a relationship's REVIEWED status was never conferred by any
review process -- it was a constant in a source file, and ranking trusted it.
Production's single PATTERN_FORMULA edge is exactly that shape: no reviewer, no
review event, no version, status REVIEWED.

Neither object type had a version, a content hash, an evidence set beyond one
optional scalar FK, or a semantic identity that means anything outside one
database. A future xerbs-core attestation has to say "a human approved THIS
object at THIS version with THIS content and THIS evidence", and none of those
four things could be named.

What this migration does
------------------------
Additive only. It adds semantic identity, version, content/evidence digests,
governance provenance and an attestation seam to ``clinical_relationship`` and
``safety_rule``; the same identity and provenance columns to
``clinical_entity``; and three shared tables giving the two previously
ungoverned types the version history, evidence association and review event
stream that clinical entities have had since 0001.

What it deliberately does NOT do
--------------------------------
*   It does not invent semantic ids. Rows authored without one keep
    ``external_id IS NULL``. A fabricated identity would be indistinguishable
    from a real one later.
*   It does not fabricate review events, evidence, or actors. Nothing is
    written to any event table.
*   It does not promote or demote any row's ``review_status``.
*   It does not turn a historical REVIEWED into proof that a human reviewed
    anything. That is the whole point of ``governance_provenance``: the
    backfill classifies each row from the records that exist, and a row
    holding REVIEWED with no review event behind it is labelled
    LEGACY_UNREVIEWED -- which is a statement about provenance, not a grant of
    authority.

Backfill rules, deterministic and derived only from existing rows:

    entity/relationship/safety rule is not REVIEWED   -> UNREVIEWED
    REVIEWED, no review event at all                  -> LEGACY_UNREVIEWED
    REVIEWED, approver == submitter                   -> LEGACY_SELF_REVIEWED
    REVIEWED, approver != submitter                   -> LEGACY_INDEPENDENTLY_REVIEWED

Relationships and safety rules have no event stream before this migration, so
every REVIEWED row among them necessarily lands on LEGACY_UNREVIEWED. That is
not a heuristic -- it is the literal truth about how they got their status.

Existing ``clinical_relationship.source_id`` / ``safety_rule.source_id`` values
are copied into ``governed_object_source`` so the evidence that WAS recorded is
preserved. Nothing is added that was not already there, and no evidence tier is
invented -- this schema has no tier column and this migration does not create
one.

Ranking impact
--------------
``LEGACY_ELIGIBILITY_GRANDFATHERED`` in app/services/governance/lifecycle.py is
True, so legacy rows keep the eligibility they already had and this migration
changes no live retrieval behaviour. Flipping it to False is AI-GOV2-D, and it
will make every pre-attestation REVIEWED relationship non-ranking-eligible
until a human re-reviews it. That is a real capability change and belongs to a
phase that schedules it, not to this one.

Downgrade
---------
Drops the three new tables and the added columns. It is a schema downgrade
only: the governance classification it recorded is lost, which is acceptable
because the classification is fully re-derivable from the same rows by running
the upgrade again. Application rollback is separate -- code at 0007 ignores
these columns entirely, except that it would once again create relationships
as REVIEWED.
"""

import sqlalchemy as sa
from alembic import op

revision = '0008_x1d_gov2c1'
down_revision = '0007_x1d_auditorder1'
branch_labels = None
depends_on = None


UNREVIEWED = 'UNREVIEWED'
LEGACY_UNREVIEWED = 'LEGACY_UNREVIEWED'
LEGACY_SELF_REVIEWED = 'LEGACY_SELF_REVIEWED'
LEGACY_INDEPENDENTLY_REVIEWED = 'LEGACY_INDEPENDENTLY_REVIEWED'


def _governed_columns(table, id_len=600):
    """The identical column set both governed types receive."""
    op.add_column(table, sa.Column('external_id', sa.String(length=id_len), nullable=True))
    op.add_column(table, sa.Column('version', sa.Integer(), nullable=False, server_default='1'))
    op.add_column(table, sa.Column('content_hash', sa.String(length=64), nullable=True))
    op.add_column(table, sa.Column('evidence_hash', sa.String(length=64), nullable=True))
    op.add_column(table, sa.Column('governance_provenance', sa.String(length=40), nullable=True))
    op.add_column(table, sa.Column('review_attestation_id', sa.String(length=64), nullable=True))
    op.add_column(table, sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column(table, sa.Column('submitter_subject', sa.String(length=128), nullable=True))
    op.add_column(table, sa.Column('last_material_editor_subject', sa.String(length=128), nullable=True))
    op.add_column(table, sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column(table, sa.Column('retired_at', sa.DateTime(timezone=True), nullable=True))
    # A unique INDEX, not a unique CONSTRAINT. SQLite cannot ALTER a
    # constraint onto an existing table (alembic would need batch
    # copy-and-move), and the test suite runs on SQLite. A unique index
    # gives the identical guarantee on both dialects with plain DDL.
    op.create_index('uq_%s_external_id' % table, table, ['external_id'], unique=True)
    op.create_index('ix_%s_governance_provenance' % table, table, ['governance_provenance'])


def upgrade():
    # ---- clinical_entity: semantic identity + provenance -------------------
    op.add_column('clinical_entity', sa.Column('external_id', sa.String(length=255), nullable=True))
    op.add_column('clinical_entity', sa.Column('governance_provenance', sa.String(length=40), nullable=True))
    op.create_index('uq_clinical_entity_external_id', 'clinical_entity',
                    ['external_id'], unique=True)
    op.create_index('ix_clinical_entity_governance_provenance', 'clinical_entity', ['governance_provenance'])

    # ---- the two previously ungoverned types -------------------------------
    _governed_columns('clinical_relationship')
    _governed_columns('safety_rule')

    # ---- shared version / evidence / event tables --------------------------
    op.create_table(
        'governed_object_version',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('object_type', sa.String(length=40), nullable=False),
        sa.Column('object_id', sa.String(length=64), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('external_id', sa.String(length=600), nullable=True),
        sa.Column('snapshot', sa.JSON(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=True),
        sa.Column('evidence_hash', sa.String(length=64), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint('object_type', 'object_id', 'version',
                            name='uq_governed_object_version'),
    )
    op.create_index('ix_governed_object_version_object', 'governed_object_version',
                    ['object_type', 'object_id'])

    op.create_table(
        'governed_object_source',
        sa.Column('object_type', sa.String(length=40), primary_key=True),
        sa.Column('object_id', sa.String(length=64), primary_key=True),
        sa.Column('source_id', sa.String(length=128),
                  sa.ForeignKey('source_registry.source_id', ondelete='RESTRICT'),
                  primary_key=True),
        sa.Column('source_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('locator', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    op.create_table(
        'governed_object_review_event',
        sa.Column('event_id', sa.String(length=64), primary_key=True),
        sa.Column('object_type', sa.String(length=40), nullable=False),
        sa.Column('object_id', sa.String(length=64), nullable=False),
        sa.Column('action', sa.String(length=40), nullable=False),
        sa.Column('actor_subject', sa.String(length=128), nullable=False),
        sa.Column('from_status', sa.String(length=20), nullable=True),
        sa.Column('to_status', sa.String(length=20), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('attestation_id', sa.String(length=64), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint('object_type', 'object_id', 'version',
                            name='uq_governed_object_review_event_version'),
    )
    op.create_index('ix_governed_object_review_event_object',
                    'governed_object_review_event', ['object_type', 'object_id'])

    conn = op.get_bind()

    # ---- backfill: entity provenance, from the event stream ----------------
    # Classified only from rows that exist. No inference, no default-to-good.
    conn.execute(sa.text("""
        UPDATE clinical_entity SET governance_provenance = :unrev
         WHERE review_status <> 'REVIEWED'
    """), {"unrev": UNREVIEWED})

    conn.execute(sa.text("""
        UPDATE clinical_entity SET governance_provenance = :legacy_none
         WHERE review_status = 'REVIEWED'
           AND NOT EXISTS (SELECT 1 FROM review_event re
                            WHERE re.entity_id = clinical_entity.id
                              AND re.action = 'APPROVED')
    """), {"legacy_none": LEGACY_UNREVIEWED})

    # approver vs submitter, both taken from the recorded events
    rows = conn.execute(sa.text("""
        SELECT e.id,
               (SELECT re.actor_id FROM review_event re
                 WHERE re.entity_id = e.id AND re.action = 'APPROVED'
                 ORDER BY re.version DESC LIMIT 1) AS approver,
               (SELECT re.actor_id FROM review_event re
                 WHERE re.entity_id = e.id AND re.action = 'SUBMITTED_FOR_REVIEW'
                 ORDER BY re.version DESC LIMIT 1) AS submitter
          FROM clinical_entity e
         WHERE e.review_status = 'REVIEWED'
           AND e.governance_provenance IS NULL
    """)).fetchall()
    for entity_id, approver, submitter in rows:
        value = (LEGACY_SELF_REVIEWED
                 if approver is not None and approver == submitter
                 else LEGACY_INDEPENDENTLY_REVIEWED)
        conn.execute(
            sa.text("UPDATE clinical_entity SET governance_provenance = :v "
                    "WHERE id = :i"), {"v": value, "i": entity_id})

    # ---- backfill: relationships and safety rules --------------------------
    # Neither type has ever had a review event, so every REVIEWED row among
    # them is LEGACY_UNREVIEWED by construction, not by guess.
    for table in ('clinical_relationship', 'safety_rule'):
        conn.execute(sa.text(
            "UPDATE %s SET governance_provenance = :unrev "
            "WHERE review_status <> 'REVIEWED'" % table), {"unrev": UNREVIEWED})
        conn.execute(sa.text(
            "UPDATE %s SET governance_provenance = :legacy_none "
            "WHERE review_status = 'REVIEWED'" % table),
            {"legacy_none": LEGACY_UNREVIEWED})

    # ---- preserve the evidence that WAS recorded ---------------------------
    # The scalar source_id becomes a row in the association table. Nothing is
    # added that did not already exist; rows without a source stay without one,
    # and a relationship with no evidence cannot become eligible.
    conn.execute(sa.text("""
        INSERT INTO governed_object_source
               (object_type, object_id, source_id, source_version, locator, created_at)
        SELECT 'CLINICAL_RELATIONSHIP', r.id, r.source_id, 1, NULL, CURRENT_TIMESTAMP
          FROM clinical_relationship r
         WHERE r.source_id IS NOT NULL
    """))
    conn.execute(sa.text("""
        INSERT INTO governed_object_source
               (object_type, object_id, source_id, source_version, locator, created_at)
        SELECT 'SAFETY_RULE', x.id, x.source_id, 1, NULL, CURRENT_TIMESTAMP
          FROM safety_rule x
         WHERE x.source_id IS NOT NULL
    """))

    # Deliberately absent: any INSERT into review_event,
    # governed_object_review_event or governed_object_version. This migration
    # observed history; it did not create any.


def downgrade():
    op.drop_index('ix_governed_object_review_event_object',
                  table_name='governed_object_review_event')
    op.drop_table('governed_object_review_event')
    op.drop_table('governed_object_source')
    op.drop_index('ix_governed_object_version_object',
                  table_name='governed_object_version')
    op.drop_table('governed_object_version')

    for table in ('safety_rule', 'clinical_relationship'):
        op.drop_index('ix_%s_governance_provenance' % table, table_name=table)
        op.drop_index('uq_%s_external_id' % table, table_name=table)
        for col in ('retired_at', 'updated_at', 'last_material_editor_subject',
                    'submitter_subject', 'reviewed_at', 'review_attestation_id',
                    'governance_provenance', 'evidence_hash', 'content_hash',
                    'version', 'external_id'):
            op.drop_column(table, col)

    op.drop_index('ix_clinical_entity_governance_provenance',
                  table_name='clinical_entity')
    op.drop_index('uq_clinical_entity_external_id', table_name='clinical_entity')
    op.drop_column('clinical_entity', 'governance_provenance')
    op.drop_column('clinical_entity', 'external_id')
