from flask import Blueprint, render_template, request, flash, jsonify, url_for, redirect
from flask_login import login_required, current_user

from .ml_model.model_metrics import GroundVisualizer
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


@views.route('/create_class_distribution_plot', methods=['POST'])
@login_required
def create_class_distribution_plot():
    return redirect(url_for('views.class_distribution_plot'))


@views.route('/class_distribution_plot', methods=['GET'])
@login_required
def class_distribution_plot():
    gv = GroundVisualizer()
    gv.load_data()
    image_path = gv.plot_class_distribution()
    print(image_path)
    return render_template("class_distribution_plot.html", image_path=url_for('static', filename=image_path),
                           user=current_user)


@views.route('/create_sample_spectra_plot', methods=['POST'])
@login_required
def create_sample_spectra_plot():
    return redirect(url_for('views.sample_spectra_plot'))


@views.route('/sample_spectra_plot', methods=['GET'])
@login_required
def sample_spectra_plot():
    gv = GroundVisualizer()
    gv.load_data()
    image_path = gv.plot_sample_spectra()
    print(image_path)
    return render_template("sample_spectra_plot.html", image_path=url_for('static', filename=image_path),
                           user=current_user)


@views.route('/create_comprehensive_review_plot', methods=['POST'])
@login_required
def create_comprehensive_review_plot():
    return redirect(url_for('views.comprehensive_review_plot'))


@views.route('/comprehensive_review_plot', methods=['GET'])
@login_required
def comprehensive_review_plot():
    gv = GroundVisualizer()
    gv.load_data()
    image_path = gv.plot_comprehensive_review()
    print(image_path)
    return render_template("comprehensive_review_plot.html", image_path=url_for('static', filename=image_path),
                           user=current_user)
