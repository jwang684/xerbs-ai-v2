"""x1d-attest1 entity attestation reference, and a corrected event constraint

Why 0009 exists rather than an edit to 0008
-------------------------------------------
0008 is a committed migration artifact and is frozen. It gave the attestation
seam (``review_attestation_id``, ``reviewed_at``) to ``clinical_relationship``
and ``safety_rule`` only, because at that point clinical entity approval was
still a local, request-body-authorised path and there was nothing to point at.

X1D-AIV2-ATTEST1 closes that path: a clinical entity now reaches REVIEWED only
through a verified xerbs-core attestation. The implementation proved the gap
immediately -- ``_apply_approval`` set ``review_attestation_id`` on a
ClinicalEntity, SQLAlchemy accepted it as an ordinary Python attribute because
no such column existed, and the reference was silently dropped on commit. A
test caught it; the honest fix is the column, forward.

Additive. No backfill: no entity has ever been approved by attestation, so
every existing row correctly has NULL here. Nothing is promoted, demoted or
reclassified, and no review event is created.

Second correction: governed_object_review_event uniqueness
----------------------------------------------------------
0008 put ``UNIQUE(object_type, object_id, version)`` on the event table,
copying the clinical-entity convention where every transition bumps the
version so the pair is naturally unique. Relationships and safety rules do not
work that way: CREATED and SUBMITTED_FOR_REVIEW can legitimately sit at the
same version, and the constraint made the second one impossible -- the
attested chain could not even be submitted.

Bumping the version on approval instead was considered and rejected: the
attestation binds an exact version, so bumping it at approval time would
invalidate the attestation that authorised it.

The uniqueness that is actually true is per ACTION: one CREATED, one
SUBMITTED_FOR_REVIEW, one APPROVED_BY_ATTESTATION per version. Replay of an
approval is caught before the insert by an explicit idempotency check, not by
this constraint.

Downgrade restores the 0008 shape and drops the entity columns. Code at 0008
ignores the columns; it would, however, hit the old constraint again.
"""

import sqlalchemy as sa
from alembic import op

revision = '0009_x1d_attest1'
down_revision = '0008_x1d_gov2c1'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('clinical_entity',
                  sa.Column('review_attestation_id', sa.String(length=64), nullable=True))
    op.add_column('clinical_entity',
                  sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True))

    # Uniqueness per (object, version, ACTION). Implemented as a unique index
    # for the same reason 0008 did: SQLite cannot ALTER a constraint onto an
    # existing table, and the suite runs on SQLite.
    with op.batch_alter_table('governed_object_review_event') as batch:
        batch.drop_constraint('uq_governed_object_review_event_version',
                              type_='unique')
    op.create_index('uq_governed_object_review_event_action',
                    'governed_object_review_event',
                    ['object_type', 'object_id', 'version', 'action'],
                    unique=True)


def downgrade():
    op.drop_index('uq_governed_object_review_event_action',
                  table_name='governed_object_review_event')
    with op.batch_alter_table('governed_object_review_event') as batch:
        batch.create_unique_constraint('uq_governed_object_review_event_version',
                                       ['object_type', 'object_id', 'version'])
    op.drop_column('clinical_entity', 'reviewed_at')
    op.drop_column('clinical_entity', 'review_attestation_id')
