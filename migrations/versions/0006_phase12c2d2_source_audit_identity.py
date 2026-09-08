"""phase12c2d2 structured source identity on audit_event

Adds audit_event.source_id so Source governance events carry canonical Source
identity in their own column. audit_event.entity_id is deliberately NOT widened
and never reused for Source ids: source_id is unvalidated client-supplied text
and could otherwise collide with the clinical entity namespace.

Existing audit_event rows stay valid (source_id is nullable and defaults NULL).

Revision ID: 0006_phase12c2d2
Revises: 0005_phase12c2d1
"""
from alembic import op
import sqlalchemy as sa

revision = '0006_phase12c2d2'
down_revision = '0005_phase12c2d1'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('audit_event', sa.Column('source_id', sa.String(length=128), nullable=True))
    op.create_index('ix_audit_event_source_id', 'audit_event', ['source_id'])


def downgrade():
    op.drop_index('ix_audit_event_source_id', table_name='audit_event')
    op.drop_column('audit_event', 'source_id')
