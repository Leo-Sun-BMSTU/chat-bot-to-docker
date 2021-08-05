from sklearn.model_selection import train_test_split
from carcass import Carcass
from classifier import Classifier
from estimator import Estimator
from importer import Importer

def check_the_lab(dir1, dir2, dir3):
    imp_df_carcass = Importer()
    df_carcass = imp_df_carcass.getFile(dir1)
    carcass = Carcass()
    carcass.parseRawData(df_carcass.data)
    carcass.normalize()
    X_train_carcass, X_test_carcass, y_train_carcass, y_test_carcass = train_test_split(carcass._X, carcass._y,
                                                                                        test_size=0.3, random_state=11)

    # Import data processed by student. Without normalizing. Train test split.
    imp_df_student = Importer()
    df_student = imp_df_student.getFile(dir2)
    carcass_student = Carcass()
    carcass_student.parseRawData(df_student.data)
    X_train_student, X_test_student, y_train_student, y_test_student = train_test_split(carcass_student._X,
                                                                                        carcass_student._y,
                                                                                        test_size=0.3, random_state=11)

    # Set general metric that student specified.
    generalMetric = "accuracy"
    command = "1"

    # Import pickle. Parse clf from pickle. Calculate all metrics using data processed by student.
    # Save and Display general metric only
    print("Student classifier: ")
    imp_pickle_student = Importer()
    userData = imp_pickle_student.getFile(dir3)
    classifier_student = Classifier()
    classifier_student.clf, classifier_student.lab = classifier_student.get_clf_from_pickle(userData.data)
    print(classifier_student.lab)
    classifier_student.calculate_metrics(y_test_student, classifier_student.clf.predict(X_test_student))
    classifier_student.get_general_metric(generalMetric)

    # Create classifier similar with pickle. Fit using carcass data. Calculate all metrics using carcass data.
    # Save and Display general metric only
    print("Carcass classifier: ")
    classifier_carcass = Classifier()
    classifier_carcass._clf = classifier_student._clf.__class__(solver='liblinear')
    classifier_carcass._clf.fit(X_train_carcass, y_train_carcass)
    classifier_carcass.calculate_metrics(y_test_carcass, classifier_carcass._clf.predict(X_test_carcass))
    classifier_carcass.get_general_metric(generalMetric)

    # Create Estimator. Compare metric
    estimator = Estimator(classifier_student._generalMetric, classifier_carcass._generalMetric, classifier_student.lab,
                          command)
    estimator.compare_metric()
