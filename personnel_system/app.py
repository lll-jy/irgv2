"""
Personnel Information Management System

A Flask-based web application for managing personnel information with role-based access control.
Features:
- User authentication (login/logout)
- User registration and profile management  
- Role-based permissions (user, admin, super_admin)
- Personnel information CRUD operations
- Admin dashboard for user management
"""

import os
import secrets
from functools import wraps
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash


# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///personnel.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录 / Please login first'


# Role constants
class Role:
    USER = 'user'
    ADMIN = 'admin'
    SUPER_ADMIN = 'super_admin'


# Database Models
class User(UserMixin, db.Model):
    """User model for authentication and authorization."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default=Role.USER, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to PersonnelInfo
    personnel_info = db.relationship('PersonnelInfo', backref='user', uselist=False, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set the password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches the hash."""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Check if user has admin privileges."""
        return self.role in [Role.ADMIN, Role.SUPER_ADMIN]
    
    def is_super_admin(self):
        """Check if user is super admin."""
        return self.role == Role.SUPER_ADMIN


class PersonnelInfo(db.Model):
    """Personnel information model."""
    __tablename__ = 'personnel_info'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    department = db.Column(db.String(100))
    position = db.Column(db.String(100))
    hire_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    return db.session.get(User, int(user_id))


# Decorators for role-based access control
def admin_required(f):
    """Decorator to require admin privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('需要管理员权限 / Admin access required', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


def super_admin_required(f):
    """Decorator to require super admin privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_super_admin():
            flash('需要超级管理员权限 / Super admin access required', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


# Routes
@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not username or not email or not password:
            flash('请填写所有必填字段 / Please fill all required fields', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('密码不匹配 / Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('密码至少6个字符 / Password must be at least 6 characters', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('用户名已存在 / Username already exists', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('邮箱已被注册 / Email already registered', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        # First user becomes super admin
        if User.query.count() == 0:
            user.role = Role.SUPER_ADMIN
        
        db.session.add(user)
        db.session.commit()
        
        flash('注册成功，请登录 / Registration successful, please login', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('登录成功 / Login successful', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        
        flash('用户名或密码错误 / Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    flash('已退出登录 / Logged out successfully', 'success')
    return redirect(url_for('index'))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """View and edit user profile and personnel info."""
    personnel = PersonnelInfo.query.filter_by(user_id=current_user.id).first()
    
    if request.method == 'POST':
        # Update personnel info
        if not personnel:
            personnel = PersonnelInfo(user_id=current_user.id)
            db.session.add(personnel)
        
        personnel.full_name = request.form.get('full_name', '').strip()
        personnel.phone = request.form.get('phone', '').strip()
        personnel.address = request.form.get('address', '').strip()
        personnel.department = request.form.get('department', '').strip()
        personnel.position = request.form.get('position', '').strip()
        personnel.notes = request.form.get('notes', '').strip()
        
        hire_date_str = request.form.get('hire_date', '').strip()
        if hire_date_str:
            try:
                personnel.hire_date = datetime.strptime(hire_date_str, '%Y-%m-%d').date()
            except ValueError:
                pass
        
        db.session.commit()
        flash('信息已更新 / Information updated', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', personnel=personnel)


@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard - view all users."""
    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)


@app.route('/admin/user/<int:user_id>')
@login_required
@admin_required
def admin_view_user(user_id):
    """Admin view user details."""
    user = db.session.get(User, user_id)
    if not user:
        flash('用户不存在 / User not found', 'error')
        return redirect(url_for('admin_dashboard'))
    
    personnel = PersonnelInfo.query.filter_by(user_id=user_id).first()
    return render_template('admin_view_user.html', user=user, personnel=personnel)


@app.route('/admin/user/<int:user_id>/role', methods=['POST'])
@login_required
@super_admin_required
def admin_change_role(user_id):
    """Super admin can change user roles."""
    user = db.session.get(User, user_id)
    if not user:
        flash('用户不存在 / User not found', 'error')
        return redirect(url_for('admin_dashboard'))
    
    if user.id == current_user.id:
        flash('不能修改自己的角色 / Cannot change your own role', 'error')
        return redirect(url_for('admin_view_user', user_id=user_id))
    
    new_role = request.form.get('role')
    if new_role in [Role.USER, Role.ADMIN, Role.SUPER_ADMIN]:
        user.role = new_role
        db.session.commit()
        flash('角色已更新 / Role updated', 'success')
    else:
        flash('无效的角色 / Invalid role', 'error')
    
    return redirect(url_for('admin_view_user', user_id=user_id))


@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
@super_admin_required
def admin_delete_user(user_id):
    """Super admin can delete users."""
    user = db.session.get(User, user_id)
    if not user:
        flash('用户不存在 / User not found', 'error')
        return redirect(url_for('admin_dashboard'))
    
    if user.id == current_user.id:
        flash('不能删除自己的账号 / Cannot delete your own account', 'error')
        return redirect(url_for('admin_view_user', user_id=user_id))
    
    db.session.delete(user)
    db.session.commit()
    flash('用户已删除 / User deleted', 'success')
    return redirect(url_for('admin_dashboard'))


def init_db():
    """Initialize the database."""
    with app.app_context():
        db.create_all()


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
