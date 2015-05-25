import nose
import sys


if nose.run(argv=['', 'menpo3d']):
    sys.exit(0)
else:
    sys.exit(1)
