import inspect
import metaworld

print('metaworld version', getattr(metaworld, '__version__', 'unknown'))
print('ML1 signature', inspect.signature(metaworld.ML1))
try:
    ml1 = metaworld.ML1('pick_place')
    print('train classes keys', list(ml1.train_classes.keys())[:10])
except Exception as exc:
    print('pick_place error', repr(exc))

try:
    ml1 = metaworld.ML1('pick-place')
    print('pick-place train classes', list(ml1.train_classes.keys())[:10])
except Exception as exc:
    print('pick-place error', repr(exc))
