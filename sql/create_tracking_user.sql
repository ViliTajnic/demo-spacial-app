WHENEVER SQLERROR EXIT SQL.SQLCODE

PROMPT Creating dedicated Oracle user/schema: TRACKING

DECLARE
  user_count NUMBER := 0;
BEGIN
  SELECT COUNT(*)
    INTO user_count
    FROM dba_users
   WHERE username = 'TRACKING';

  IF user_count = 0 THEN
    EXECUTE IMMEDIATE q'[CREATE USER tracking IDENTIFIED BY "TrackingPwd123"]';
  END IF;
END;
/

ALTER USER tracking IDENTIFIED BY "TrackingPwd123";
GRANT CREATE SESSION TO tracking;
GRANT CREATE TABLE TO tracking;
GRANT CREATE VIEW TO tracking;
GRANT CREATE SEQUENCE TO tracking;
ALTER USER tracking QUOTA UNLIMITED ON USERS;

PROMPT User TRACKING is ready.
PROMPT Next: connect as TRACKING and run sql/schema_oracle_26ai.sql
