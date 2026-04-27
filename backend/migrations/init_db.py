"""
Database migration helper using Alembic.
Run:  python -m backend.migrations.init_db
"""
import asyncio
from app.database import engine, Base
from app.models.crypto import CandleRecord, PredictionRecord  # noqa: F401 – registers models


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("All tables created.")


if __name__ == "__main__":
    asyncio.run(create_tables())
