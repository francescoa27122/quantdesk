"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Created: ${create_date}

Why this change:
    <!-- The defect or need. A migration with no rationale is unreviewable. -->

Downgrade safety:
    <!-- State plainly whether downgrade loses data. If it does, say what. -->
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | None = ${repr(branch_labels)}
depends_on: str | None = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
