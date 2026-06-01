from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# ───────── FILES ─────────
USER_FILE = "users.json"
CONTACT_FILE = "contacts.json"

# create files if not exist
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(CONTACT_FILE):
    with open(CONTACT_FILE, "w") as f:
        json.dump([], f)

# ───────── PAGES ─────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/register-page')
def register_page():
    return render_template('register.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/holiday')
def holiday():
    return render_template('holiday.html')
@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/press_release')
def press_release():
    return render_template('press_release.html')

# ───────── PATTERN PAGES ─────────

@app.route('/reversal_chart_pattern')
def reversal_chart_pattern():
    return render_template('reversal_chart_pattern.html')

@app.route('/sidewase_chart_pattern')
def sideways_chart_pattern():
    return render_template('sidewase_chart_pattern.html')

@app.route('/single_candlestick')
def single_candlestick():
    return render_template('single_candlestick.html')

@app.route('/double_candelestic')
def double_candlestick():
    return render_template('double_candelestic.html')

@app.route('/triple_candelestic')
def triple_candlestick():
    return render_template('triple_candelestic.html')

@app.route('/continuation_chart_pattern')
def continuation_chart_pattern():
    return render_template('continuation_chart_pattern.html')



# ───────── AUTH ─────────
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    with open(USER_FILE, "r") as f:
        users = json.load(f)

    for user in users:
        if user["username"] == username:
            return jsonify({"success": False, "message": "User already exists"})

    users.append({"username": username, "password": password})

    with open(USER_FILE, "w") as f:
        json.dump(users, f)

    return jsonify({"success": True, "message": "Registered successfully"})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    with open(USER_FILE, "r") as f:
        users = json.load(f)

    for user in users:
        if user["username"] == username and user["password"] == password:
            return jsonify({"success": True, "message": "Login successful"})

    return jsonify({"success": False, "message": "Invalid credentials"})


# ───────── CONTACT ─────────
@app.route('/contact-submit', methods=['POST'])
def contact_submit():
    data = request.get_json()

    with open(CONTACT_FILE, "r") as f:
        contacts = json.load(f)

    contacts.append(data)

    with open(CONTACT_FILE, "w") as f:
        json.dump(contacts, f)

    return jsonify({"success": True, "message": "Message saved successfully"})


# ───────── RUN ─────────
if __name__ == '__main__':
    app.run(debug=True, port=8080)