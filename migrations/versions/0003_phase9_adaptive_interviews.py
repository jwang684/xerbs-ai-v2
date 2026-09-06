"""phase9 adaptive diagnostic interviews
Revision ID: 0003_phase9_interviews
Revises: 0002_phase7_safety
"""
from alembic import op
import sqlalchemy as sa

revision='0003_phase9_interviews'
down_revision='0002_phase7_safety'
branch_labels=None
depends_on=None

def upgrade():
    op.create_table('diagnostic_interview',
        sa.Column('id',sa.String(64),primary_key=True),
        sa.Column('status',sa.String(40),nullable=False),
        sa.Column('intake_snapshot',sa.JSON(),nullable=False),
        sa.Column('reasoning_snapshot',sa.JSON(),nullable=False),
        sa.Column('current_turn',sa.Integer(),nullable=False),
        sa.Column('convergence_score',sa.Float(),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),nullable=False),
        sa.Column('updated_at',sa.DateTime(timezone=True),nullable=False))
    op.create_index('ix_diagnostic_interview_status','diagnostic_interview',['status'])
    op.create_table('diagnostic_interview_turn',
        sa.Column('id',sa.String(64),primary_key=True),
        sa.Column('interview_id',sa.String(64),sa.ForeignKey('diagnostic_interview.id',ondelete='CASCADE'),nullable=False),
        sa.Column('turn_number',sa.Integer(),nullable=False),
        sa.Column('answers',sa.JSON(),nullable=False),
        sa.Column('asked_questions',sa.JSON(),nullable=False),
        sa.Column('reasoning_snapshot',sa.JSON(),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),nullable=False),
        sa.UniqueConstraint('interview_id','turn_number',name='uq_diagnostic_interview_turn'))
    op.create_index('ix_diagnostic_interview_turn_interview_id','diagnostic_interview_turn',['interview_id'])

def downgrade():
    op.drop_table('diagnostic_interview_turn')
    op.drop_table('diagnostic_interview')
