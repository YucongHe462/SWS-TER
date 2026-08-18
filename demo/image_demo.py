"""Run SWS-TER inference with the bundled legacy MMRotate API."""

import argparse

from mmrotate.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--out-file', default='result.jpg')
    args = parser.parse_args()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    result = inference_detector(model, args.image)
    show_result_pyplot(model, args.image, result,
                       score_thr=args.score_thr,
                       out_file=args.out_file)


if __name__ == '__main__':
    main()

