"""phase12c2d1 source governance schema foundation

Adds governance columns to source_registry, version pinning and a citation
locator to entity_source, and a dedicated source_review_event table.

No behavioral change: no governance transitions, no eligibility change, no
SourceType enum, no bibliographic fields, no source version-history table.

Legacy rows migrate to review_status='DRAFT' and version=1. No existing Source
ever becomes REVIEWED automatically.

audit_event is intentionally left unchanged; see the Phase 12C-2D1 report.

Revision ID: 0005_phase12c2d1
Revises: 0004_phase10
"""
from alembic import op
import sqlalchemy as sa

revision = '0005_phase12c2d1'
down_revision = '0004_phase10'
branch_labels = None
depends_on = None


def upgrade():
    # --- source_registry governance columns -------------------------------
    # Constant server_defaults keep this a plain ADD COLUMN on SQLite, so the
    # table is never rebuilt and inbound foreign keys (entity_source,
    # clinical_relationship, safety_rule) stay intact.
    op.add_column('source_registry', sa.Column('review_status', sa.String(length=20), nullable=False, server_default='DRAFT'))
    op.add_column('source_registry', sa.Column('version', sa.Integer(), nullable=False, server_default='1'))
    op.add_column('source_registry', sa.Column('created_by', sa.String(length=255), nullable=True))
    op.add_column('source_registry', sa.Column('reviewed_by', sa.String(length=255), nullable=True))
    op.add_column('source_registry', sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index('ix_source_registry_review_status', 'source_registry', ['review_status'])

    # Backfill updated_at for legacy rows from their creation time.
    op.execute('UPDATE source_registry SET updated_at = created_at WHERE updated_at IS NULL')

    # --- entity_source version pinning + citation locator -----------------
    op.add_column('entity_source', sa.Column('source_version', sa.Integer(), nullable=False, server_default='1'))
    op.add_column('entity_source', sa.Column('locator', sa.Text(), nullable=True))

    # --- source governance trail ------------------------------------------
    op.create_table(
        'source_review_event',
        sa.Column('event_id', sa.String(length=64), primary_key=True),
        sa.Column('source_id', sa.String(length=128), sa.ForeignKey('source_registry.source_id', ondelete='RESTRICT'), nullable=False),
        sa.Column('action', sa.String(length=80), nullable=False),
        sa.Column('actor_id', sa.String(length=255), nullable=False),
        sa.Column('actor_role', sa.String(length=80), nullable=True),
        sa.Column('from_status', sa.String(length=20), nullable=True),
        sa.Column('to_status', sa.String(length=20), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_source_review_event_source_id', 'source_review_event', ['source_id'])


def downgrade():
    op.drop_index('ix_source_review_event_source_id', table_name='source_review_event')
    op.drop_table('source_review_event')
    op.drop_column('entity_source', 'locator')
    op.drop_column('entity_source', 'source_version')
    op.drop_index('ix_source_registry_review_status', table_name='source_registry')
    op.drop_column('source_registry', 'updated_at')
    op.drop_column('source_registry', 'reviewed_by')
    op.drop_column('source_registry', 'created_by')
    op.drop_column('source_registry', 'version')
    op.drop_column('source_registry', 'review_status')
