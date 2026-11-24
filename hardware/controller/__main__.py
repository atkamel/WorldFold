"""Print controller states as JSON lines or capture USB reports as hex."""

import argparse
from dataclasses import asdict
from itertools import islice
import json
import sys

from .controller import ControllerError, read_controller
from .gip import parse_input_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--raw", action="store_true", help="print one hex line per USB read")
    modes.add_argument("--decode", metavar="HEX", help="decode one captured report without USB")
    parser.add_argument("--count", type=int, help="exit after printing this many live reports")
    args = parser.parse_args()
    if args.count is not None and (args.count < 1 or args.decode is not None):
        parser.error("--count requires a positive count and live USB mode")
    if args.decode is not None:
        try:
            report = parse_input_report(bytes.fromhex(args.decode))
        except ValueError as exc:
            parser.error(str(exc))
        if report is None:
            parser.error("expected a GIP input or Guide report")
        print(json.dumps(asdict(report.state())))
        return 0

    try:
        with read_controller() as stream:
            print("Reading PowerA controller. Press Ctrl-C to stop.", file=sys.stderr)
            if args.raw:
                lines = (reading.raw.hex(" ") for reading in stream)
            else:
                lines = (json.dumps(asdict(reading.state)) for reading in stream if reading.state is not None)
            for line in islice(lines, args.count):
                print(line, flush=True)
    except KeyboardInterrupt:
        return 130
    except ControllerError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
