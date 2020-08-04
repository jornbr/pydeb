import argparse
from matplotlib import pyplot
from . import infer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('species')
    parser.add_argument('--c_T', type=float, default=None)
    parser.add_argument('--t', type=float, default=None)
    args = parser.parse_args()
    model = infer.get_model_by_name(args.species)
    if args.c_T is None:
        T = infer.get_typical_temperature(model.col_id)
        args.c_T = model.get_temperature_correction(T)
        print('Body temperature: %.2f degrees Celsius' % (T,))
    model.describe(c_T=args.c_T)
    delta_t = 0.1
    result = model.simulate(int((args.t or max(model.a_99, model.a_m) / args.c_T) / delta_t), delta_t, c_T=args.c_T)
    fig = model.plot_result(**result, c_T=args.c_T)
    pyplot.show()

if __name__ == '__main__':
    main()