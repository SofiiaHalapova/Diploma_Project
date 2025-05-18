import os
import random
import re
import shutil
import string

from flask import Blueprint, render_template, url_for, redirect, send_file
from flask import request, flash, session, json, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from .ml_model.k_nearest_classes_prediction import GeneratePredictionKNearest
from .ml_model.logistic_regression_classes_prediction import GeneratePredictionLogisticRegression
from .ml_model.neural_network_classes_prediction import GeneratePredictionCNN
from .ml_model.random_forest_classes_prediction import GeneratePredictionRandomForest
from .ml_model.svm_classes_prediction import GeneratePredictionSVM
from .ml_model.xg_boost_classes_prediction import GeneratePredictionXGBoost
from .ml_model.data_evaluation import DataEvaluation
from .models import Project, db

views = Blueprint('views', __name__)

ALLOWED_EXTENSIONS = {'zip', 'tar', 'gz'}
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(CURRENT_DIR, 'static')
UPLOAD_DATASETS_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')


def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def get_classification_classes(raw_input: str) -> str:
    cleaned = re.sub(r'[,;|/\\]+', ' ', raw_input)

    cleaned = re.sub(r'\W+', ' ', cleaned)

    words = cleaned.lower().split()
    unique_words = list(dict.fromkeys(words))

    return ' '.join(unique_words)


def load_dataset_file(upload_method, dataset_file, dataset_link, error, project_name):
    kagglehub_pattern = re.compile(r"kagglehub\.dataset_download\(\"([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)\"\)")
    dataset = None

    if upload_method == "link" and dataset_link:
        if kagglehub_pattern.match(dataset_link):
            dataset = dataset_link
        else:
            flash("Invalid dataset link type. Download allowed from kagglehub", category='error')
            error = True
    elif upload_method == "archive" and dataset_file and allowed_file(dataset_file.filename, ALLOWED_EXTENSIONS):
        filename = secure_filename(dataset_file.filename)
        file_path = os.path.join(UPLOAD_DATASETS_FOLDER, filename)
        print("views archive file path", file_path)
        dataset_file.save(file_path)
        dataset = unarchive_dataset(file_path, project_name)
    else:
        flash("Invalid dataset file type. Allowed: zip, tar, gz.", category='error')
        error = True

    return dataset, error


def validate_upload_input(project_name, classification_classes,
                          classification_method, dataset_file, dataset_link):
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
    return error


def get_current_project_or_redirect():
    project_id = session.get('current_project_id')
    if not project_id:
        flash("No current project selected.", 'error')
        return None, redirect(url_for("views.home"))

    project = Project.query.get(project_id)
    if not project:
        flash("Project not found.", 'error')
        return None, redirect(url_for("views.home"))

    return project, None


def generate_classes_dictionary(class_names) -> dict:
    class_names = class_names.split()
    class_names_dict = {}
    num = 0
    for i in class_names:
        class_names_dict.update({num: i})
        num += 1

    return class_names_dict


def render_plot_page(plot_func, template, plot_name):
    project, redirect_response = get_current_project_or_redirect()
    if redirect_response:
        return redirect_response

    class_names_dict = generate_classes_dictionary(project.classification_classes)

    de = DataEvaluation(class_names=class_names_dict)
    de.load_data(project.dataset)

    img_directory = os.path.join(os.path.abspath(STATIC_FOLDER), 'imgs')

    os.makedirs(img_directory, exist_ok=True)

    random_string = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5))

    full_image_path = os.path.join(img_directory, f'{plot_name}_{random_string}.png')

    image_path = plot_func(de, save_path=full_image_path)

    return render_template(template, image_path=url_for('static', filename=image_path), user=current_user,
                           project=project)


def unarchive_dataset(file_path, project_name):
    extract_dir = os.path.join(UPLOAD_DATASETS_FOLDER, f"{secure_filename(project_name)}")
    os.makedirs(extract_dir, exist_ok=True)

    if file_path.endswith(".zip"):
        shutil.unpack_archive(file_path, extract_dir, "zip")
    elif file_path.endswith(".tar"):
        shutil.unpack_archive(file_path, extract_dir, "tar")
    elif file_path.endswith(".gz"):
        shutil.unpack_archive(file_path, extract_dir, "gztar")
    else:
        raise ValueError("Unsupported archive format")

    return extract_dir


# main page of the website
@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)


@views.route('/api/project_count')
@login_required
def count_projects():
    count = Project.query.count()
    return jsonify({'count': count})


@views.route('/new_project', methods=['GET', 'POST'])
@login_required
def new_project():
    os.makedirs(UPLOAD_DATASETS_FOLDER, exist_ok=True)

    if request.method == "POST":
        form = request.form
        dataset = None

        project_name = form.get('project_name')
        classification_classes = get_classification_classes(form.get('classification_classes'))
        classification_method = form.get('classification_method')
        upload_method = form.get('upload_method')
        dataset_link = form.get('dataset_link')
        dataset_file = request.files.get('dataset_archive')

        error = validate_upload_input(project_name, classification_classes,
                                      classification_method, dataset_file, dataset_link)

        if not error:
            dataset, error = load_dataset_file(upload_method, dataset_file, dataset_link, error, project_name)

        if not error:
            project = Project(
                project_name=project_name,
                classification_method=classification_method,
                classification_classes=classification_classes,
                dataset=dataset,
                user_id=current_user.id
            )

            db.session.add(project)
            db.session.commit()

            session['current_project_id'] = project.id

            flash("Project has been successfully created.", category='success')

            return redirect(url_for("views.project_details", project_id=project.id))

    return render_template("home.html", user=current_user)


@views.route('/project/<int:project_id>', methods=['GET', 'POST'])
@login_required
def project_details(project_id):
    project = Project.query.get_or_404(project_id)
    session['current_project_id'] = project_id
    return render_template("project_workspace.html", user=current_user, project=project,
                           class_names=generate_classes_dictionary(project.classification_classes))


@views.route('/create_class_distribution_plot', methods=['POST'])
@login_required
def create_class_distribution_plot():
    return redirect(url_for('views.class_distribution_plot'))


@views.route('/class_distribution_plot', methods=['GET'])
@login_required
def class_distribution_plot():
    return render_plot_page(lambda de, save_path: de.plot_class_distribution(save_path=save_path),
                            "class_distribution_plot.html", plot_name='class_distribution_plot')


@views.route('/create_sample_spectra_plot', methods=['GET', 'POST'])
@login_required
def create_sample_spectra_plot():
    if request.method == "POST":
        spectras_to_analyze = request.form.getlist('multiDropdownSpectras')
        session['spectras_to_analyze'] = spectras_to_analyze
    return redirect(url_for('views.sample_spectra_plot'))


@views.route('/sample_spectra_plot', methods=['GET', 'POST'])
@login_required
def sample_spectra_plot():
    selected_classes = session.pop('spectras_to_analyze', None)
    return render_plot_page(lambda de, save_path: de.plot_sample_spectra(save_path=save_path,
                                                                         selected_classes=selected_classes),
                            "sample_spectra_plot.html", plot_name='sample_spectra_plot')


@views.route('/create_comprehensive_review_plot', methods=['POST'])
@login_required
def create_comprehensive_review_plot():
    return redirect(url_for('views.comprehensive_review_plot'))


@views.route('/comprehensive_review_plot', methods=['GET'])
@login_required
def comprehensive_review_plot():
    return render_plot_page(lambda de, save_path: de.plot_comprehensive_review(save_path=save_path),
                            "comprehensive_review_plot.html", plot_name="comprehensive_review_plot")


@views.route('/create_classes_prediction_plot', methods=['GET', 'POST'])
@login_required
def create_classes_prediction_plot():
    print("create_classes_prediction_plot is called")
    return render_template("loading.html", user=current_user)


@views.route('/classes_prediction_plot', methods=['GET'])
@login_required
def classes_prediction_plot():
    print("classes_prediction_plot is called")
    project, redirect_response = get_current_project_or_redirect()
    if redirect_response:
        return redirect_response

    class_names_dict = generate_classes_dictionary(project.classification_classes)
    gp = None

    if project.classification_method == "random-forest-classifier":
        gp = GeneratePredictionRandomForest(class_names=class_names_dict)
        print("data is loading Random Forest")
    elif project.classification_method == "xg-boost":
        gp = GeneratePredictionXGBoost(class_names=class_names_dict)
        print("data is loading xg-boost")
    elif project.classification_method == "k-nearest":
        gp = GeneratePredictionKNearest(class_names=class_names_dict)
        print("data is loading k-nearest")
    elif project.classification_method == "logistic-regression":
        gp = GeneratePredictionLogisticRegression(class_names=class_names_dict)
        print("data is loading logistic-regression")
    elif project.classification_method == "svm":
        gp = GeneratePredictionSVM(class_names=class_names_dict)
        print("data is loading svm")
    else:
        gp = GeneratePredictionCNN(class_names=class_names_dict)
        print("data is loading cnn")

    gp.load_data(project.dataset)
    image_path = gp.compute_classification_results()

    return render_template("class_prediction_plot.html",
                           image_path=url_for('static', filename=image_path), user=current_user,
                           project=project, classification_method=project.classification_method)


@views.route('/download', methods=['GET', 'POST'])
def download():
    project, redirect_response = get_current_project_or_redirect()
    if redirect_response:
        return redirect_response

    filename = request.args.get("filename")

    if not filename:
        flash("Filename not provided", "error")
        return redirect(url_for("views.project_details", project_id=project.id))

    save_path = os.path.join(STATIC_FOLDER, filename.replace("static/", "").replace("\\", "/"))

    if not os.path.exists(save_path):
        flash("File not found", "error")
        return redirect(url_for("views.project_details", project_id=project.id))

    return send_file(save_path, as_attachment=True)


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
