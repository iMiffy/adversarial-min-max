import os
import torch
import torch.nn as nn
from torchvision import transforms
from util import ImageFolderWithPaths, deepfool, UnNormalize, clip_tensor, str2bool
from PIL import Image
import argparse
import warnings
warnings.filterwarnings("ignore")


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--py-version", required = True,
                help = "only Python 3.6 or 3.7 can work on this algorithm")
ap.add_argument("-m", "--mode", required=True, help = "min or max L2 norm")
ap.add_argument("-d", "--data-path", default = '../data', help = "path to input images")
ap.add_argument("-p", "--model-path", default = '../model/model.pt',
                help = "path to model used for inferencing")
ap.add_argument("-s", "--save", type = str2bool, default = 't',
                help = "save image or not")
ap.add_argument("-o", "--output-path", help = "Path to the output image (optional)")
args = vars(ap.parse_args())


if args['py_version'] == str(3.7):
    from load_model_37 import load_model

elif args['py_version'] == str(3.6):
    from load_model_36 import load_model

else:
    raise SystemError('Only can run on Python 3.6 or 3.7')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(args['model_path'], device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transformation = transforms.Compose([transforms.Resize(128),
                                     transforms.CenterCrop(128),
                                     transforms.ToTensor(),
                                     normalize,])
test_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(args['data_path'], transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                normalize,])), batch_size=1, num_workers=0)


success = 0
unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
new_clip = lambda x: clip_tensor(x, 0, 1)

for data, label, path in test_loader:
    print('Image:', path[0])

    data = data.to(device)
    label = label.to(device)
    print('Initital label:', label.item())

    if args['mode'] == 'min':
        # Deepfool adversarial attack
        r, loop_i, label_orig, label_pert, pert_image = deepfool(data.squeeze(0), model, num_classes=4)

    elif args['mode'] == 'max':
        # Iterative-Fast Gradient Sign Method adversarial attack
        ce_loss = nn.CrossEntropyLoss()
        data.requires_grad = True
        for i in range(500):
            data.grad = None
            output = model(data)
            loss = ce_loss(output, label)
            loss.backward()

            adv_noise = 0.01 * torch.sign(data.grad.data)
            data.data = data.data + adv_noise

        output = model(data)
        _ = output.argmax(dim=1, keepdim=True)
        label_pert = _.item()
        pert_image = data.detach()
        label_orig = label

    else:
        raise SystemError('Select either min or max mode')

    print('Label after attacking:', label_pert)
    success += 1 if label_orig != label_pert else 0


    if args['save']:
        _ = unnorm(pert_image.cpu().squeeze(0))
        _ = transforms.Lambda(new_clip)(_)
        _ = _.permute(1,2,0) * 255
        image = _.numpy().astype('uint8')
        Image.fromarray(image).save(os.path.join(args['output_path'], os.path.basename(path[0])))
    print('*********************************************')

print('Successful cases:', success)
print('Total cases:', len(test_loader))