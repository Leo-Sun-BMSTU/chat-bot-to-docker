from sklearn import preprocessing

class Carcass:
    X = property()
    y = property()

    def parseRawData(self, data):
        data.columns = ['A' + str(i) for i in range(len(data.columns) - 1)] + ['class']
        self._y = data[data.columns[-1]]
        self._X = data[data.columns[:-1]]

    def normalize(self):
        global categorical_columns, numerical_columns, nonbinary_columns, binary_columns
        le = preprocessing.LabelEncoder()

        # duplicated dropping
        nunique = self._X.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        self._X = self._X.drop(cols_to_drop, axis=1)

        # column types split
        for c in self._X.columns:
            if self._X[c].value_counts().count() == 2:
                self._X[c] = le.fit_transform(self._X[c])
                categorical_columns = [c for c in self._X.columns if self._X[c].dtype.name == 'object']
                numerical_columns = [c for c in self._X.columns if
                                     (self._X[c].dtype.name != 'object') and (self._X[c].value_counts().count() > 20)]
                binary_columns = [c for c in self._X.columns if self._X[c].value_counts().count() == 2]
                nonbinary_columns = [c for c in self._X.columns if
                                     (self._X[c].value_counts().count() > 2) and (
                                                 self._X[c].value_counts().count() < 20) and (
                                             c not in categorical_columns)]

        self._X[numerical_columns] = self._X[numerical_columns].fillna(self._X[numerical_columns].median(axis=0),
                                                                       axis=0)

        def get_count(count):
            return counter.get(count, np.nan)

        for i in categorical_columns:
            counter = Counter(self._X[i])
            self._X.loc[:, i] = self._X[i].apply(get_count)

        for c in binary_columns:
            self._X[c] = self._X[c].fillna(0.5)

        for c in nonbinary_columns:
            self._X[c] = self._X[c].fillna(self._X.mode().iloc[0])

        return self._X

    @X.getter
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @X.deleter
    def X(self):
        del self._X

    @y.getter
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @y.deleter
    def y(self):
        del self._y
