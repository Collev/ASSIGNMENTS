-- 1. Drop an index named IdxPhone from customers table
DROP INDEX IdxPhone ON customers;

-- 2. Create a user named bob with password, restricted to localhost
CREATE USER 'bob'@'localhost' IDENTIFIED BY 'S$cu3r3!';

-- 3. Grant INSERT privilege to user bob on the salesDB database
GRANT INSERT ON salesDB.* TO 'bob'@'localhost';
