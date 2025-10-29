import re
from typing import Any
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.tools import Tool


class SecureSQLQueryExecutor:
    """DML 방지 및 쿼리 검증을 수행하는 안전한 SQL 실행기"""

    # DML 키워드 정의
    DML_KEYWORDS = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "REPLACE",
        "MERGE",
        "EXEC",
        "EXECUTE",
    ]

    def __init__(self, db: SQLDatabase):
        self.db = db

    def validate_query(self, query: str) -> tuple[bool, str]:
        """
        쿼리가 안전한지 검증
        """

        query_upper = query.upper()

        # 1. DML 키워드 체크
        for keyword in self.DML_KEYWORDS:
            pattern = r"\b" + keyword + r"\b"

            if re.search(pattern, query_upper):
                return (
                    False,
                    f"🚫 Security Error: {keyword} statements are not allowed!",
                )

        # 2. SQL Injection 방지
        statments = [s.strip() for s in query.split(";") if s.strip()]
        if len(statments) > 1:
            return False, "🚫 Security Error: Multiple statements are not allowed!"

        # 3. 주석을 이용한 우회 시도 체크
        if "--" in query or "/*" in query or "*/" in query:
            return False, "🚫 Security Error: Comments in queries are not allowed!"

        return True, ""

    def execute(self, query: str) -> str:
        """검증 후 쿼리 실행"""
        # 검증
        is_valid, error_msg = self.validate_query(query)
        if not is_valid:
            return error_msg

        try:
            # 실행
            result = self.db.run(query)
            return result
        except Exception as e:
            return f"❌ Query execution error: {str(e)}"


def create_secure_sql_tools(db: SQLDatabase, llm: Any) -> list:
    """보안 강화된 SQL 도구들"""
    secure_executor = SecureSQLQueryExecutor(db)

    # 기존 toolkit의 도구들을 가져오되, query 도구만 교체
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    secure_tools = []

    for tool in tools:
        if tool.name == "sql_db_query":
            secure_query_tool = Tool(
                name="sql_db_query",
                description="""
                Execute a SQL query against the database and get back the result.
                If the query is not correct, an error message will be returned.
                If an error is returned, rewrite the query, check the query, and try again.
                """,
                func=secure_executor.execute,
            )
            secure_tools.append(secure_query_tool)
        else:
            secure_tools.append(tool)

    return secure_tools
