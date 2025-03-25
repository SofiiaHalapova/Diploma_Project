from flask import Blueprint, render_template, request, flash, jsonify, url_for, redirect
from flask_login import login_required, current_user
from .models import Note, db
import json

views = Blueprint('views', __name__)


# main page of the website
@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)


@views.route('/create_project', methods=['GET', 'POST'])
@login_required
def create_new_project():
    return redirect(url_for("views.new_project"))


@views.route('/new_project', methods=['GET', 'POST'])
@login_required
def new_project():
    return render_template("new_project.html", user=current_user)
