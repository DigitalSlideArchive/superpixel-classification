import sys
import time
from xml.sax.saxutils import escape


class ProgressHelper:
    def __init__(self, name, comment='', report=True):
        """
        A name is required.  The comment is shown in the popup as commentary
        on the task.
        """
        self._name = name
        self._comment = comment
        self._start = time.time()
        self._last = 0
        self._report = report

    def __enter__(self):
        if self._report:
            print("""<filter-start>
<filter-name>%s</filter-name>
<filter-comment>%s</filter-comment>
</filter-start>""" % (escape(self._name), escape(self._comment)))
            sys.stdout.flush()
        self._start = time.time()
        return self

    def items(self, items):
        self._items = items
        self._item_progress = []

    def item_progress(self, item, val):
        if item not in self._items:
            return
        idx = self._items.index(item)
        if len(self._item_progress) < len(self._items):
            self._item_progress += [0] * (len(self._items) - len(self._item_progress))
        self._item_progress[idx] = val
        total = sum(self._item_progress) / len(self._item_progress)
        self.progress(total)

    def progress(self, val):
        if self._report and (val == 0 or val == 1 or time.time() - self._last >= 0.1):
            print("""<filter-progress>%s</filter-progress>""" % val)
            sys.stdout.flush()
            self._last = time.time()

    def message(self, comment):
        self._comment = comment
        if self._report:
            print("""<filter-comment>%s</filter-comment>""" % escape(comment))
            sys.stdout.flush()

    def name(self, name):
        # Leave the original name alone
        if self._report:
            print("""<filter-name>%s</filter-name>""" % escape(name))
            sys.stdout.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        duration = end - self._start
        if self._report:
            print("""<filter-end>
 <filter-name>%s</filter-name>
 <filter-time>%s</filter-time>
</filter-end>""" % (escape(self._name), duration))
            sys.stdout.flush()


__all__ = ['ProgressHelper']
