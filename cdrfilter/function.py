import re
from abc import ABC, abstractmethod

from cdrstr import to_camel_case


def year_to_age(year):
    import datetime

    ny = datetime.datetime.now().year
    return ny - year


class FilterFunction(ABC):
    @abstractmethod
    def map(self, e):
        pass


class SplitYearAsAge(FilterFunction):
    def map(self, e):
        res = e.copy()
        title = res['title']
        year, title = re.split(r'\s+', title, maxsplit=1)
        res['title'] = title

        res['age'] = year_to_age(int(year))

        return res


class SplitBrand(FilterFunction):
    def map(self, e):
        res = e.copy()
        title = res['title']
        brand, title = re.split(r'\s+', title, maxsplit=1)
        res['brand'] = brand.lower()
        res['title'] = title.lower()

        return res


class ParseMileage(FilterFunction):
    def map(self, e):
        res = e.copy()

        if 'mileage' not in res:
            res['mileage'] = 0
            return res

        mileage = res['mileage']
        mileage, _ = re.split(r'\s+', mileage)
        res['mileage'] = int(mileage.replace(',', ''))

        return res


class ParsePrice(FilterFunction):
    def map(self, e):
        res = e.copy()
        price = res['primaryPrice']
        price = re.sub(r'[$,]', '', price)
        try:
            res['price'] = int(price)
        except ValueError:
            res['price'] = 0
        res.pop('primaryPrice')

        return res


class RemoveLoanPrice(FilterFunction):
    def map(self, e):
        res = e.copy()
        res.pop('loanPrice')

        return res


class RemoveStockType(FilterFunction):
    def map(self, e):
        res = e.copy()
        res.pop('stockType')

        return res


class StandardizeBadges(FilterFunction):
    def map(self, e):
        def clear(text):
            t = text
            t = re.split(r'\|', t)[0]
            t = to_camel_case(t)
            return t

        res = e.copy()
        badges = res['badges']
        badges = [clear(b) for b in badges]
        res['badges'] = badges

        return res


class DealRatingsFromBadges(FilterFunction):
    def __init__(self, deal_ratings):
        self._deal_ratings = deal_ratings.copy()

    def map(self, e):
        res = e.copy()
        badges = res['badges']
        deal_ratings = []
        other_badges = []
        for b in badges:
            if b in self._deal_ratings:
                deal_ratings.append(b)
            else:
                other_badges.append(b)
        res['dealRatings'] = deal_ratings
        res['badges'] = other_badges

        return res


class OneHotEncodeDealRatings(FilterFunction):
    def __init__(self, deal_ratings):
        self._deal_ratings = deal_ratings

    def map(self, e):
        res = e.copy()
        for b in self._deal_ratings:
            res[f'dealRating_{to_camel_case(b)}'] = 1 if b in res['dealRatings'] else 0

        res.pop('dealRatings')

        return res


class OneHotEncodeBadges(FilterFunction):
    def __init__(self, badges):
        self._badges = badges

    def map(self, e):
        res = e.copy()
        for b in self._badges:
            res[f'badge_{to_camel_case(b)}'] = 1 if b in res['badges'] else 0

        res.pop('badges')

        return res


class ClearModelNames(FilterFunction):
    def map(self, e):
        res = e.copy()
        title = res['title']
        title = re.sub(r'[!\"#$%&()*+.,-/:;=?@\[\\\]^_`{|}~]', '', title)
        title = to_camel_case(title)
        res['model'] = title
        res.pop('title')

        return res
