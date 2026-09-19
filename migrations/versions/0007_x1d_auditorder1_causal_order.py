"""x1d-auditorder1 causal ordering for governance events

Governance history was ordered by (created_at, event_id). created_at comes from
a clock that advances every 1-2ms, and event_id is aud-/srcevt-/evt- followed by
a random uuid, so whenever two transitions of one aggregate landed in the same
tick the tiebreaker was a coin toss. With three events that is a 1-in-6 chance
of the correct order; the audit reproduced it 11 times out of 12 with the
timestamp held constant. A clinical governance trail could report
SOURCE_APPROVED before SOURCE_CREATED, with nothing to indicate it had.

The causal key already existed. _transition raises VERSION_CONFLICT, bumps
row.version by exactly one, and stamps the event with it, so version IS the
lifecycle position. review_event and source_review_event already store it;
audit_event stored it only inside the JSON payload.

This migration therefore:

  * adds audit_event.version, nullable, and backfills it EXACTLY from
    payload->>'version' for aggregate-scoped rows. Nothing is inferred. Rows
    that belong to no aggregate -- provider telemetry, clarification rejection
    records, gap records -- keep NULL, because they have no causal position and
    no ordered query can return them;

  * adds the three uniqueness constraints that turn "version is unique per
    aggregate" from a convention the code happened to honour into something the
    database refuses to break. That second part is not cosmetic: the optimistic
    check in _transition reads without a row lock, and the audit demonstrated
    two concurrent writers both committing the same version. After this, the
    second one fails.

Applying this to a database that already contains duplicate (aggregate,
version) pairs will fail at constraint creation, by design -- a duplicate means
two events claim one lifecycle position and no migration can know which came
first. Run the pre-flight first.

Revision ID: 0007_x1d_auditorder1
Revises: 0006_phase12c2d2
"""
from alembic import op
import sqlalchemy as sa

revision = '0007_x1d_auditorder1'
down_revision = '0006_phase12c2d2'
branch_labels = None
depends_on = None


# Aggregate-scoped means the row names a source or a clinical entity. Only those
# have a lifecycle to be ordered against.
_SCOPED = "(source_id IS NOT NULL OR entity_id IS NOT NULL)"

# Set-based rather than a row-by-row pass in Python: the backfill reads a value
# that is already recorded and writes it to a column, and doing that one row at
# a time would hold the migration open for as long as the table is large. Two
# dialects are in play and both are explicit here; anything else stops rather
# than guessing at its JSON syntax.
_BACKFILL = {
    "postgresql": (
        "UPDATE audit_event SET version = (payload->>'version')::int "
        "WHERE version IS NULL AND " + _SCOPED + " "
        "AND payload->>'version' ~ '^[0-9]+$'"),
    "sqlite": (
        "UPDATE audit_event SET version = "
        "CAST(json_extract(payload, '$.version') AS INTEGER) "
        "WHERE version IS NULL AND " + _SCOPED + " "
        "AND json_extract(payload, '$.version') IS NOT NULL"),
}

# Reads back what was written and compares it to the payload it came from. A
# backfill that claims to be exact should be able to say so out loud.
_VERIFY = {
    "postgresql": (
        "SELECT count(*) FROM audit_event WHERE " + _SCOPED + " "
        "AND payload->>'version' ~ '^[0-9]+$' "
        "AND (version IS NULL OR version <> (payload->>'version')::int)"),
    "sqlite": (
        "SELECT count(*) FROM audit_event WHERE " + _SCOPED + " "
        "AND json_extract(payload, '$.version') IS NOT NULL "
        "AND (version IS NULL OR version <> "
        "CAST(json_extract(payload, '$.version') AS INTEGER))"),
}


def _dialect(bind):
    name = bind.dialect.name
    if name not in _BACKFILL:
        raise RuntimeError(
            "X1D-AUDITORDER1 has no JSON backfill defined for dialect %r. "
            "Add one explicitly rather than letting the column stay NULL." % name)
    return name


def upgrade():
    bind = op.get_bind()
    name = _dialect(bind)

    op.add_column('audit_event', sa.Column('version', sa.Integer(), nullable=True))

    bind.exec_driver_sql(_BACKFILL[name])
    mismatched = bind.exec_driver_sql(_VERIFY[name]).scalar()
    if mismatched:
        raise RuntimeError(
            "X1D-AUDITORDER1 backfill disagrees with payload version on %d "
            "audit_event row(s); refusing to add the uniqueness constraints."
            % mismatched)

    # SQLite cannot ALTER TABLE ADD CONSTRAINT, so alembic rebuilds the table.
    # PostgreSQL adds it in place, which is what staging and production do.
    if name == "sqlite":
        with op.batch_alter_table('audit_event') as b:
            b.create_unique_constraint('uq_audit_event_source_version',
                                       ['source_id', 'version'])
            b.create_unique_constraint('uq_audit_event_entity_version',
                                       ['entity_id', 'version'])
        with op.batch_alter_table('source_review_event') as b:
            b.create_unique_constraint('uq_source_review_event_source_version',
                                       ['source_id', 'version'])
        with op.batch_alter_table('review_event') as b:
            b.create_unique_constraint('uq_review_event_entity_version',
                                       ['entity_id', 'version'])
    else:
        op.create_unique_constraint('uq_audit_event_source_version',
                                    'audit_event', ['source_id', 'version'])
        op.create_unique_constraint('uq_audit_event_entity_version',
                                    'audit_event', ['entity_id', 'version'])
        op.create_unique_constraint('uq_source_review_event_source_version',
                                    'source_review_event', ['source_id', 'version'])
        op.create_unique_constraint('uq_review_event_entity_version',
                                    'review_event', ['entity_id', 'version'])


def downgrade():
    """Drops the constraints and the column. No recorded fact is destroyed:
    every value in audit_event.version was copied from payload->>'version',
    which this migration never touched and leaves in place."""
    bind = op.get_bind()
    name = bind.dialect.name

    if name == "sqlite":
        with op.batch_alter_table('review_event') as b:
            b.drop_constraint('uq_review_event_entity_version', type_='unique')
        with op.batch_alter_table('source_review_event') as b:
            b.drop_constraint('uq_source_review_event_source_version', type_='unique')
        with op.batch_alter_table('audit_event') as b:
            b.drop_constraint('uq_audit_event_entity_version', type_='unique')
            b.drop_constraint('uq_audit_event_source_version', type_='unique')
            b.drop_column('version')
    else:
        op.drop_constraint('uq_review_event_entity_version',
                           'review_event', type_='unique')
        op.drop_constraint('uq_source_review_event_source_version',
                           'source_review_event', type_='unique')
        op.drop_constraint('uq_audit_event_entity_version',
                           'audit_event', type_='unique')
        op.drop_constraint('uq_audit_event_source_version',
                           'audit_event', type_='unique')
        op.drop_column('audit_event', 'version')
