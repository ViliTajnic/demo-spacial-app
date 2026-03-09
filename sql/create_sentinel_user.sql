WHENEVER SQLERROR EXIT SQL.SQLCODE

PROMPT Creating dedicated Oracle user/schema: SENTINEL

DECLARE
  user_count NUMBER := 0;
BEGIN
  SELECT COUNT(*)
    INTO user_count
    FROM dba_users
   WHERE username = 'SENTINEL';

  IF user_count = 0 THEN
    EXECUTE IMMEDIATE q'[CREATE USER sentinel IDENTIFIED BY "SentinelPwd123"]';
  END IF;
END;
/

ALTER USER sentinel IDENTIFIED BY "SentinelPwd123";
GRANT CREATE SESSION TO sentinel;
GRANT CREATE TABLE TO sentinel;
GRANT CREATE VIEW TO sentinel;
GRANT CREATE SEQUENCE TO sentinel;
ALTER USER sentinel QUOTA UNLIMITED ON USERS;

PROMPT User SENTINEL is ready.
PROMPT Next: connect as SENTINEL and run sql/schema_oracle_26ai.sql
