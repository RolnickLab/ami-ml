import sqlalchemy as sa
import sqlalchemy.orm


DATABASE_URL = "postgresql+psycopg2://def-drolnick-ami-owner@localhost:5432/def-drolnick-ami"
DATABASE_SCHEMA_NAMESPACE="fieldguide"  # "fieldguide", "gbif"

engine = sa.create_engine(
    DATABASE_URL,
    connect_args={'options': f'-csearch_path={DATABASE_SCHEMA_NAMESPACE}'},
    # echo=True,
    future= False,  # pandas does not support the 2.0 syntax yet
)

Session = sa.orm.sessionmaker(engine)


def test_query():
    with Session.begin() as session:
        # query with text-only sql statements
        statement = sa.text("select * from test")
        results = session.execute(statement)
        for item in results:
            print(item.id, item.name)

    with Session.begin() as session:
        # query with text statements and python parameters
        statement = sa.text("select * from test where id <= :max")
        results = session.execute(statement, {"max": 2})
        for item in results:
            print(item.id, item.name)

    with Session.begin() as session:
        # query with python using a "TableClause"
        t = sa.table("test", sa.column("id"), sa.column("name"), schema="sandbox")
        statement = sa.select(t).where(t.c.name.ilike('%butt%'))
        results = session.execute(statement)
        for item in results:
            print(item.id, item.name)

if __name__ == "__main__":
    test_query()