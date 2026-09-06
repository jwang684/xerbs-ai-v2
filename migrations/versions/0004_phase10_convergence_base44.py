"""phase10 convergence and Base44 generation contract

Revision ID: 0004_phase10
Revises: 0003_phase9_interviews
"""
from alembic import op
import sqlalchemy as sa

revision = '0004_phase10'
down_revision = '0003_phase9_interviews'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('diagnostic_interview', sa.Column('convergence_snapshot', sa.JSON(), nullable=True))
    op.create_table(
        'generation_request',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('idempotency_key', sa.String(length=255), nullable=False),
        sa.Column('correlation_id', sa.String(length=255), nullable=False),
        sa.Column('external_request_id', sa.String(length=255), nullable=True),
        sa.Column('organization_id', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('request_hash', sa.String(length=64), nullable=False),
        sa.Column('request_snapshot', sa.JSON(), nullable=False),
        sa.Column('response_snapshot', sa.JSON(), nullable=True),
        sa.Column('error_code', sa.String(length=80), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('idempotency_key', name='uq_generation_request_idempotency_key'),
    )
    op.create_index('ix_generation_request_idempotency_key','generation_request',['idempotency_key'],unique=True)
    op.create_index('ix_generation_request_correlation_id','generation_request',['correlation_id'])
    op.create_index('ix_generation_request_external_request_id','generation_request',['external_request_id'])
    op.create_index('ix_generation_request_organization_id','generation_request',['organization_id'])
    op.create_index('ix_generation_request_status','generation_request',['status'])

def downgrade():
    op.drop_index('ix_generation_request_status', table_name='generation_request')
    op.drop_index('ix_generation_request_organization_id', table_name='generation_request')
    op.drop_index('ix_generation_request_external_request_id', table_name='generation_request')
    op.drop_index('ix_generation_request_correlation_id', table_name='generation_request')
    op.drop_index('ix_generation_request_idempotency_key', table_name='generation_request')
    op.drop_table('generation_request')
    op.drop_column('diagnostic_interview','convergence_snapshot')
