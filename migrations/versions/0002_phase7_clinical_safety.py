"""phase7 clinical safety and relationships

Revision ID: 0002_phase7_safety
Revises: 0001_phase6_persistent_clinical_corpus
"""
from alembic import op
import sqlalchemy as sa

revision='0002_phase7_safety'
down_revision='0001_phase6'
branch_labels=None
depends_on=None

def upgrade():
    op.create_table('clinical_relationship',
        sa.Column('id',sa.String(64),primary_key=True),
        sa.Column('source_entity_id',sa.String(64),sa.ForeignKey('clinical_entity.id',ondelete='CASCADE'),nullable=False),
        sa.Column('target_entity_id',sa.String(64),sa.ForeignKey('clinical_entity.id',ondelete='CASCADE'),nullable=False),
        sa.Column('relationship_type',sa.String(40),nullable=False),
        sa.Column('review_status',sa.String(20),nullable=False),
        sa.Column('source_id',sa.String(128),sa.ForeignKey('source_registry.source_id',ondelete='RESTRICT'),nullable=True),
        sa.Column('created_by',sa.String(255),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),nullable=False),
        sa.UniqueConstraint('source_entity_id','target_entity_id','relationship_type',name='uq_clinical_relationship'))
    op.create_index('ix_clinical_relationship_source_entity_id','clinical_relationship',['source_entity_id'])
    op.create_index('ix_clinical_relationship_target_entity_id','clinical_relationship',['target_entity_id'])
    op.create_index('ix_clinical_relationship_relationship_type','clinical_relationship',['relationship_type'])
    op.create_index('ix_clinical_relationship_review_status','clinical_relationship',['review_status'])
    op.create_table('safety_rule',
        sa.Column('id',sa.String(64),primary_key=True),
        sa.Column('target_entity_id',sa.String(64),sa.ForeignKey('clinical_entity.id',ondelete='CASCADE'),nullable=False),
        sa.Column('rule_type',sa.String(40),nullable=False),
        sa.Column('trigger_term',sa.String(255),nullable=False),
        sa.Column('severity',sa.String(20),nullable=False),
        sa.Column('action',sa.String(20),nullable=False),
        sa.Column('message',sa.Text(),nullable=False),
        sa.Column('review_status',sa.String(20),nullable=False),
        sa.Column('source_id',sa.String(128),sa.ForeignKey('source_registry.source_id',ondelete='RESTRICT'),nullable=True),
        sa.Column('created_by',sa.String(255),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),nullable=False))
    for c in ['target_entity_id','rule_type','trigger_term','review_status']:
        op.create_index(f'ix_safety_rule_{c}','safety_rule',[c])

def downgrade():
    op.drop_table('safety_rule')
    op.drop_table('clinical_relationship')
