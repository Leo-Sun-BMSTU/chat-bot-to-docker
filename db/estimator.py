class Estimator:
    result = property()

    def __init__(self, metric_student, metric_carcass, lab, command):
        self._metric_student = metric_student
        self._metric_carcass = metric_carcass
        self._lab = lab
        self._command = command

    def compare_metric(self):
        if self._metric_student > self._metric_carcass:
            self._result = "1"
            print("Command #{0} passed the check for lab {1}. General metric = {2}".format(self._command,self._lab, self._metric_student))
            return self._result
        else:
            self._result = "0"
            print("Command #{0} failed the check for lab {1}. General metric = {2}".format(self._command,self._lab,self._metric_student))
            return self._result

    @result.getter
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @result.deleter
    def result(self):
        del self._result
