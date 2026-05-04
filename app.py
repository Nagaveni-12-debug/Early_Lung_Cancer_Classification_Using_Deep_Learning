from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
from werkzeug.utils import secure_filename
import sqlite3
import predict_model

app = Flask(__name__)
app.secret_key = "cancer_detection_secret_key"

# Upload folder
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}


# ============================= DATABASE INIT =============================
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


init_db()


# ============================= HELPERS =============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================= ROUTES =============================

@app.route("/")
def index():
    return render_template("index.html")


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm = request.form["confirm_password"]

        if password != confirm:
            flash("Passwords do not match!", "error")
            return redirect(url_for("register"))

        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                         (username, email, password))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except:
            flash("Username or Email already exists.", "error")
        finally:
            conn.close()

    return render_template("register.html")


# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["email"] = user["email"]

            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password!", "error")

    return render_template("login.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    username = session["username"]
    user_email = session["email"]

    conn = get_db_connection()

    total = conn.execute("SELECT COUNT(*) FROM scans WHERE user_id = ?", (user_id,)).fetchone()[0]
    cancer = conn.execute("SELECT COUNT(*) FROM scans WHERE user_id = ? AND result = 'Cancer'",
                          (user_id,)).fetchone()[0]
    healthy = conn.execute("SELECT COUNT(*) FROM scans WHERE user_id = ? AND result = 'Healthy'",
                           (user_id,)).fetchone()[0]

    recent = conn.execute(
        "SELECT id, result FROM scans WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
        (user_id,)
    ).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        user_name=username,
        user_email=user_email,
        user_id=user_id,
        total_scans=total,
        cancer_count=cancer,
        healthy_count=healthy,
        recent_scans=recent
    )


# ---------------- PROFILE ----------------
@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("profile.html",
                           username=session["username"],
                           email=session["email"])


# ---------------- PREDICT ----------------
@app.route("/predict")
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("predict.html")


# ---------------- ANALYZE ----------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"})

    file = request.files.get("file")
    if not file:
        return jsonify({"success": False, "error": "No file uploaded"})

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(save_path)

    result = predict_model.predict_image(save_path)

    if result.get("result") == "Error":
        return jsonify({"success": False, "error": result.get("message")})

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO scans (user_id, filename, result, confidence) VALUES (?, ?, ?, ?)",
        (session["user_id"], filename, result["result"], result["confidence"])
    )
    conn.commit()
    conn.close()

    session["analysis_result"] = result
    session["image_path"] = f"static/uploads/{filename}"

    return jsonify({"success": True})


# ---------------- RESULTS ----------------
@app.route("/results")
def results():
    if "user_id" not in session:
        return redirect(url_for("login"))

    result = session.get("analysis_result")
    filename = session.get("image_path")

    if not result:
        return redirect(url_for("predict"))

    return render_template(
        "results.html",
        result_text=result["result"],
        confidence=result["confidence"],
        probs=result["probs"],
        filename=filename
    )


# ---------------- HISTORY ----------------
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    scans = conn.execute(
        "SELECT * FROM scans WHERE user_id = ? ORDER BY created_at DESC",
        (session["user_id"],)
    ).fetchall()
    conn.close()

    return render_template("history.html", scans=scans)


# ============================= INFO TEMPLATE ROUTES =============================

@app.route("/prevention")
def prevention():
    return render_template("info_prevention.html")

@app.route("/understanding")
def understanding():
    return render_template("info_understanding.html")

@app.route("/symptoms")
def symptoms():
    return render_template("info_symptoms.html")

@app.route("/treatment")
def treatment():
    return render_template("info_treatment.html")

@app.route("/research")
def research():
    return render_template("info_research.html")

@app.route("/finance")
def finance():
    return render_template("resources_financial.html")

@app.route("/support")
def support():
    return render_template("resources_support.html")

@app.route("/healthtips")
def healthtips():
    return render_template("resources_healthtips.html")

@app.route("/screening")
def screening():
    return render_template("resources_screening.html")


# ---------------- ABOUT PAGES ----------------
@app.route("/contact")
def contact():
    return render_template("about_contact.html")

@app.route("/privacy")
def privacy():
    return render_template("about_privacy.html")

@app.route("/terms")
def terms():
    return render_template("about_terms.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ============================= RUN APP =============================
if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True, port=5000)
