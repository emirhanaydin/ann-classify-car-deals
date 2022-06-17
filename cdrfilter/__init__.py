class Filter:
    def __init__(self, json_array):
        self._json_array = json_array
        self._filter_fns = []

    def pipe(self, filter_fn):
        self._filter_fns.append(filter_fn)
        return self

    def apply(self):
        res = []
        for e in self._json_array:
            e_res = e.copy()
            for fn in self._filter_fns:
                e_res = fn.map(e_res)
            res.append(e_res)

        return res
