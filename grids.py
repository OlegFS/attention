from matplotlib import pyplot as plt

class GridMaker(object):
    
    def __init__(self, fig, n_rows, n_cols,
                 left=0.08, right=0.95, bottom=0.08, top=0.95,
                 wspace=0.2, hspace=0.2, **kw):
        # save attributes
        for k, v in locals().items():
            if k=='self':
                continue
            setattr(self, k, v)
        # widths/heights
        if n_cols > 1:
            self.w = (right - left - wspace) / n_cols
            self.wgap = wspace / (n_cols - 1.)
        else:
            self.w = right - left
            self.wgap = 0
        if n_rows > 1:
            self.h = (top - bottom - hspace) / n_rows
            self.hgap = hspace / (n_rows - 1.)
        else:
            self.h = top - bottom
            self.hgap = 0
        # starting x positions
        self.xi = [left]
        for c in range(1, n_cols):
            next_xi = self.xi[-1] + self.w + self.wgap
            self.xi.append(next_xi)
        # starting y positions
        self.yi = [bottom]
        for r in range(1, n_rows):
            next_yi = self.yi[-1] + self.h + self.hgap
            self.yi.append(next_yi)
        self.yi.reverse()

    def axes(self, row, col, **kw):
        if row >= self.n_rows:
            raise ValueError('row must be between 0 and %d' % self.n_rows-1)
        if col >= self.n_cols:
            raise ValueError('col must be between 0 and %d' % self.n_cols-1)
        xi = self.xi[col]
        yi = self.yi[row]
        w = self.w
        h = self.h
        ax_kws = self.kw.copy()
        ax_kws.update(**kw)
        ax = self.fig.add_axes([xi, yi, w, h], **ax_kws)
        self.fig.show()
        return ax
        
    def all_axes(self, **kw):
        axs = []
        for r in range(self.n_rows):
            row_axs = []
            for c in range(self.n_cols):
                ax = self.axes(r, c, **kw)
                row_axs.append(ax)
            axs.append(row_axs)
        return axs
        
