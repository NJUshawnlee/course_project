__author__ = 'Shawn Li'

import description
import log_norm
import mixture

description.data_description('sh000001', '20100101', '20170331')
log_norm.draw_lognorm('sh000001', '20100101', '20170331')
mixture.draw_mixture('sh000001', '20100101', '20170331')
