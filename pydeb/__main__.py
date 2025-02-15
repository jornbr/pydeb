import argparse
from . import infer, __version__

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('taxon', help='Name of the species or higher-rank taxon you want the DEB model for. For instance, "Asterias rubens" or "Mammalia"')
    parser.add_argument('--c_T', type=float, default=None, help='Temperature correction factor (dimensionless)')
    parser.add_argument('-s', '--simulate', action='store_true', help='Simulate growth, reproduction and survival and plot results')
    parser.add_argument('-t', '--time', type=float, default=None, help='Time period (number of days) to simulate')
    parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
    args = parser.parse_args()
    taxon = infer.Taxon.from_name(args.taxon)
    model = taxon.get_model()
    if args.c_T is None:
        T = taxon.typical_temperature
        args.c_T = model.get_temperature_correction(T)
        print('Body temperature: %.2f degrees Celsius' % (T,))
    model.describe(c_T=args.c_T)
    if args.simulate:
        from matplotlib import pyplot
        delta_t = 0.1
        result = model.simulate(int((args.time or max(model.a_99, model.a_m) / args.c_T) / delta_t), delta_t, c_T=args.c_T)
        fig = model.plot_result(**result, c_T=args.c_T)
        pyplot.show()

if __name__ == '__main__':
    main()