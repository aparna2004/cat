from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
from datetime import datetime, timezone, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secretkey'
DATABASE = 'toll_plaza.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            car_number TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            balance REAL NOT NULL
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            car_number TEXT NOT NULL,
            amount REAL NOT NULL,
            date TEXT DEFAULT CURRENT_DATE,
            entry_time TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(car_number) REFERENCES user(car_number)
        )''')
        conn.commit()

def get_ist_time():
    IST = timezone(timedelta(hours=5, minutes=30))  
    utc_now = datetime.now(timezone.utc) 
    return utc_now.astimezone(IST)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        car_number = request.form['car_number']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        balance = float(request.form['balance'])
        with get_db_connection() as conn:
            conn.execute("INSERT INTO user (name, car_number, email, password, balance) VALUES (?, ?, ?, ?, ?)",
                         (name, car_number, email, password, balance))
            conn.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with get_db_connection() as conn:
            user = conn.execute("SELECT * FROM user WHERE email = ?", (email,)).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user'] = user['email']
                if user['email'] == 'admin@toll.com':
                    session['role'] = 'admin' 
                    return redirect(url_for('admin'))
                else:
                    session['role'] = 'user' 
                return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        with get_db_connection() as conn:
            user = conn.execute("SELECT * FROM user WHERE email = ?", (session['user'],)).fetchone()
            transactions = conn.execute("SELECT * FROM transactions WHERE car_number = ?", (user['car_number'],)).fetchall()
        return render_template('dashboard.html', user=user, transactions=transactions)
    return redirect(url_for('login'))

@app.route('/pay_toll', methods=['POST'])
def pay_toll():
    if 'user' in session:
        amount = float(request.form['amount'])
        with get_db_connection() as conn:
            user = conn.execute("SELECT * FROM user WHERE email = ?", (session['user'],)).fetchone()
            if user['balance'] >= amount:
                new_balance = user['balance'] - amount
                entry_time = get_ist_time().strftime('%Y-%m-%d %H:%M:%S')
                conn.execute("UPDATE user SET balance = ? WHERE email = ?", (new_balance, session['user']))
                conn.execute("INSERT INTO transactions (car_number, amount, entry_time) VALUES (?, ?, ?)", 
                             (user['car_number'], amount, entry_time))
                conn.commit()
    return redirect(url_for('dashboard'))

@app.route('/recharge', methods=['POST'])
def recharge():
    if 'user' in session:
        amount = float(request.form['amount'])
        with get_db_connection() as conn:
            user = conn.execute("SELECT * FROM user WHERE email = ?", (session['user'],)).fetchone()
            new_balance = user['balance'] + amount
            conn.execute("UPDATE user SET balance = ? WHERE email = ?", (new_balance, session['user']))
            conn.commit()
    return redirect(url_for('dashboard'))

@app.route('/admin')
def admin():
    if 'user' in session and session['role'] == 'admin':
        date_filter = request.args.get('date')
        query = "SELECT * FROM transactions" + (" WHERE date = ?" if date_filter else "")
        
        with get_db_connection() as conn:
            transactions = conn.execute(query, (date_filter,) if date_filter else ()).fetchall()
        
        return render_template('admin.html', transactions=transactions)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
