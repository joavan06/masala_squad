import mysql.connector

# Connect to MySQL
connection = mysql.connector.connect(
    host="localhost",       # or "127.0.0.1"
    user="root",            # your MySQL username
    password="root",  # your MySQL password
    database="hospital"     # the database you created
)

# Check if connection is successful
if connection.is_connected():
    print("✅ Connected to MySQL database 'hospital'")

# Create a cursor to execute queries
cursor = connection.cursor()

# Example query: fetch all hospitals
cursor.execute("SELECT * FROM hospitals;")

# Fetch results
rows = cursor.fetchall()

print("\n📋 Hospital Records:")
for row in rows:
    print(row)

# Close connection
cursor.close()
connection.close()
