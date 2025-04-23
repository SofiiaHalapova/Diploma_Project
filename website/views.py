import os

from flask import Blueprint, render_template, url_for, redirect
from flask import request, flash, session, json, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from .ml_model.model_metrics import MetricsCalculator
from .models import Project, db
import re

views = Blueprint('views', __name__)

UPLOAD_DATASETS_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'zip', 'tar', 'gz'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@views.route('/api/project_count')
@login_required
def count_projects():
    count = Project.query.count()
    return jsonify({'count': count})


def get_classification_classes(raw_input: str) -> str:
    cleaned = re.sub(r'[,;|/\\]+', ' ', raw_input)

    cleaned = re.sub(r'[^\w]+', ' ', cleaned)

    words = cleaned.lower().split()
    unique_words = list(dict.fromkeys(words))

    return ' '.join(unique_words)


# main page of the website
@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)


@views.route('/new_project', methods=['GET', 'POST'])
@login_required
def new_project():
    if request.method == "POST":
        project_name = request.form.get('project_name')
        classification_classes = get_classification_classes(request.form.get('classification_classes'))
        classification_method = request.form.get('classification_method')
        upload_method = request.form.get('upload_method')
        dataset_link = request.form.get('dataset_link')
        dataset_file = request.files.get('dataset_archive')
        error = False

        if not project_name:
            flash("Project name field is required.", category='error')
            error = True
        elif Project.query.filter_by(project_name=project_name).first():
            flash("Project with such name already exists!", category='error')
            error = True
        elif not classification_classes:
            flash("Classification classes field is required.", category='error')
            error = True
        elif not classification_method:
            flash("Classification method field is required.", category='error')
            error = True
        elif not dataset_file and not dataset_link:
            flash("You must upload a dataset file or provide a link.", category='error')
            error = True
        elif dataset_file and dataset_link:
            flash("Please provide either a dataset file or a link, not both.", category='error')
            error = True

        dataset = None
        kagglehub_pattern = re.compile(r'kagglehub\.dataset_download\("([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)"\)')
        if not error:
            if upload_method == "link" and dataset_link:
                if kagglehub_pattern.match(dataset_link):
                    dataset = dataset_link
                else:
                    flash("Invalid dataset link type. Download allowed from kagglehub", category='error')
                    error = True
            elif upload_method == "archive" and dataset_file and allowed_file(dataset_file.filename):
                filename = secure_filename(dataset_file.filename)
                file_path = os.path.join(UPLOAD_DATASETS_FOLDER, filename)
                dataset_file.save(file_path)
                dataset = filename
            else:
                flash("Invalid dataset file type. Allowed: zip, tar, gz.", category='error')
                error = True

        if not error:
            current_project = Project(
                project_name=project_name,
                classification_method=classification_method,
                classification_classes=classification_classes,
                dataset=dataset,
                user_id=current_user.id
            )

            db.session.add(current_project)
            db.session.commit()

            session['current_project_id'] = current_project.id

            flash("Project has been successfully created.", category='success')

            return redirect(url_for("views.project_details", project_id=current_project.id))

    return render_template("home.html", user=current_user)


@views.route('/project/<int:project_id>', methods=['GET', 'POST'])
@login_required
def project_details(project_id):
    project = Project.query.get_or_404(project_id)
    session['current_project_id'] = project_id
    return render_template("project_workspace.html", user=current_user, project=project)


@views.route('/create_class_distribution_plot', methods=['POST'])
@login_required
def create_class_distribution_plot():
    return redirect(url_for('views.class_distribution_plot'))


@views.route('/class_distribution_plot', methods=['GET'])
@login_required
def class_distribution_plot():
    project_id = session.get('current_project_id')
    if not project_id:
        flash("No current project selected.", category='error')
        return redirect(url_for("views.home"))

    current_project = Project.query.get(project_id)
    if not current_project:
        flash("Project not found.", category='error')
        return redirect(url_for("views.home"))

    print(current_project.classification_classes)
    gv = MetricsCalculator(class_names=current_project.classification_classes)
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
    project_id = session.get('current_project_id')
    if not project_id:
        flash("No current project selected.", category='error')
        return redirect(url_for("views.home"))

    current_project = Project.query.get(project_id)
    if not current_project:
        flash("Project not found.", category='error')
        return redirect(url_for("views.home"))

    print(current_project.classification_classes)
    gv = MetricsCalculator(class_names=current_project.classification_classes)
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
    project_id = session.get('current_project_id')
    if not project_id:
        flash("No current project selected.", category='error')
        return redirect(url_for("views.home"))

    current_project = Project.query.get(project_id)
    if not current_project:
        flash("Project not found.", category='error')
        return redirect(url_for("views.home"))

    print(current_project.classification_classes)
    gv = MetricsCalculator(class_names=current_project.classification_classes)
    gv.load_data()
    image_path = gv.plot_comprehensive_review()
    print(image_path)
    return render_template("comprehensive_review_plot.html", image_path=url_for('static', filename=image_path),
                           user=current_user)


@views.route('/delete-project', methods=['POST'])
def delete_project():
    project = json.loads(request.data)
    projectId = project['projectId']
    project = Project.query.get(projectId)

    if project:
        if project.user_id == current_user.id:
            db.session.delete(project)
            db.session.commit()

    return jsonify({})