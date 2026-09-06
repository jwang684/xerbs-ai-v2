"""Phase 6 persistent clinical corpus governance tables.

Revision ID: 0001_phase6
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_phase6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("clinical_entity",
        sa.Column("id",sa.String(64),primary_key=True), sa.Column("entity_type",sa.String(20),nullable=False),
        sa.Column("name",sa.String(255),nullable=False), sa.Column("review_status",sa.String(20),nullable=False),
        sa.Column("current_version",sa.Integer(),nullable=False), sa.Column("migration_origin",sa.String(255)),
        sa.Column("created_at",sa.DateTime(timezone=True),nullable=False), sa.Column("updated_at",sa.DateTime(timezone=True),nullable=False),
        sa.Column("retired_at",sa.DateTime(timezone=True)), sa.Column("superseded_by_id",sa.String(64),sa.ForeignKey("clinical_entity.id")))
    op.create_index("ix_clinical_entity_type_status","clinical_entity",["entity_type","review_status"])
    op.create_table("source_registry", sa.Column("source_id",sa.String(128),primary_key=True),sa.Column("title",sa.String(500),nullable=False),sa.Column("citation",sa.Text()),sa.Column("url",sa.Text()),sa.Column("source_type",sa.String(80),nullable=False),sa.Column("created_at",sa.DateTime(timezone=True),nullable=False))
    op.create_table("clinical_entity_version",sa.Column("id",sa.String(64),primary_key=True),sa.Column("entity_id",sa.String(64),sa.ForeignKey("clinical_entity.id",ondelete="RESTRICT"),nullable=False),sa.Column("version",sa.Integer(),nullable=False),sa.Column("snapshot",sa.JSON(),nullable=False),sa.Column("created_by",sa.String(255),nullable=False),sa.Column("created_at",sa.DateTime(timezone=True),nullable=False),sa.UniqueConstraint("entity_id","version",name="uq_clinical_entity_version"))
    op.create_table("entity_source",sa.Column("entity_id",sa.String(64),sa.ForeignKey("clinical_entity.id",ondelete="CASCADE"),primary_key=True),sa.Column("source_id",sa.String(128),sa.ForeignKey("source_registry.source_id",ondelete="RESTRICT"),primary_key=True))
    op.create_table("ingestion_batch",sa.Column("batch_id",sa.String(64),primary_key=True),sa.Column("submitted_by",sa.String(255),nullable=False),sa.Column("source_label",sa.String(500),nullable=False),sa.Column("created_at",sa.DateTime(timezone=True),nullable=False))
    op.create_table("ingestion_item",sa.Column("id",sa.String(64),primary_key=True),sa.Column("batch_id",sa.String(64),sa.ForeignKey("ingestion_batch.batch_id",ondelete="CASCADE"),nullable=False),sa.Column("entity_id",sa.String(64),sa.ForeignKey("clinical_entity.id",ondelete="RESTRICT"),nullable=False),sa.Column("external_id",sa.String(255)),sa.Column("ordinal",sa.Integer(),nullable=False))
    op.create_table("review_event",sa.Column("event_id",sa.String(64),primary_key=True),sa.Column("entity_id",sa.String(64),sa.ForeignKey("clinical_entity.id",ondelete="RESTRICT"),nullable=False),sa.Column("entity_type",sa.String(20),nullable=False),sa.Column("action",sa.String(80),nullable=False),sa.Column("actor_id",sa.String(255),nullable=False),sa.Column("actor_role",sa.String(80)),sa.Column("from_status",sa.String(20)),sa.Column("to_status",sa.String(20)),sa.Column("version",sa.Integer(),nullable=False),sa.Column("notes",sa.Text()),sa.Column("created_at",sa.DateTime(timezone=True),nullable=False))
    op.create_table("audit_event",sa.Column("event_id",sa.String(64),primary_key=True),sa.Column("event_type",sa.String(80),nullable=False),sa.Column("entity_id",sa.String(64)),sa.Column("actor_id",sa.String(255),nullable=False),sa.Column("payload",sa.JSON(),nullable=False),sa.Column("created_at",sa.DateTime(timezone=True),nullable=False))


def downgrade():
    for table in ["audit_event","review_event","ingestion_item","ingestion_batch","entity_source","clinical_entity_version","source_registry","clinical_entity"]:
        op.drop_table(table)
