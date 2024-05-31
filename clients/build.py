from utils import get_numclasses
from utils.registry import Registry
import models

CLIENT_REGISTRY = Registry("CLIENT")
CLIENT_REGISTRY.__doc__ = """
Registry for local updater
"""

__all__ = ['get_client_type', 'get_client_type_compare']

# def get_model(args,trainset = None):
#     num_classes=get_numclasses(args,trainset)
#     print("=> Creating model '{}'".format(args.arch))
#     print("Model Option")
#     print(" 1) use_pretrained =", args.use_pretrained)
#     print(" 2) No_transfer_learning =", args.No_transfer_learning)
#     print(" 3) use_bn =", args.use_bn)
#     print(" 4) use_pre_fc =", args.use_pre_fc)
#     print(" 5) use_bn_layer =", args.use_bn_layer)
#     model = models.__dict__[args.arch](num_classes=num_classes, l2_norm=args.l2_norm, use_pretrained = args.use_pretrained, transfer_learning = not(args.No_transfer_learning), use_bn = args.use_bn, use_pre_fc = args.use_pre_fc, use_bn_layer = args.use_bn_layer)
#     return model

def get_client_type(args):
    if args.verbose:
        print(CLIENT_REGISTRY)
    print("=> Getting client type '{}'".format(args.client.type))
    client_type = CLIENT_REGISTRY.get(args.client.type)
    return client_type


def get_client_type_compare(args):
    if args.verbose:
        print(CLIENT_REGISTRY)
    print("=> Getting client_compare type '{}'".format(args.client_compare.type))
    client_type = CLIENT_REGISTRY.get(args.client_compare.type)
    return client_type

def build_client(args):
    return
