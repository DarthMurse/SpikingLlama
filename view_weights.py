import torch
import matplotlib.pyplot as plt


def main():
    teacher_pth = "out/spiking-llama-1b/teacher.pth"
    student_pth = "out/spiking-llama-1b/iter-060000-ckpt.pth"

    teacher_weights = torch.load(teacher_pth)
    student_checkpoint = torch.load(student_pth)
    student_weights = student_checkpoint['model']

    w_to_view = ['transformer.h.0.mlp.swiglu.w1.weight', 'transformer.h.0.mlp.swiglu.w2.weight', 'transformer.h.0.mlp.swiglu.w3.weight', 'transformer.h.0.attn.attn.weight', 'transformer.h.0.attn.proj.weight']
    print(teacher_weights[w_to_view[1]].shape)
    # print(student_weights[w_to_view[2]].shape)

if __name__ == "__main__":
    main()

import torch
import matplotlib.pyplot as plt

# def main():
#     teacher_pth = "out/spiking-llama-1b/teacher.pth"
#     student_pth = "out/spiking-llama-1b/iter-060000-ckpt.pth"

#     teacher_weights = torch.load(teacher_pth)
#     student_checkpoint = torch.load(student_pth)
#     student_weights = student_checkpoint['model']

#     w_to_view = ['transformer.h.0.mlp.swiglu.w1.weight', 'transformer.h.0.mlp.swiglu.w2.weight', 'transformer.h.0.mlp.swiglu.w3.weight', 'transformer.h.0.attn.attn.weight', 'transformer.h.0.attn.proj.weight']

    
#     teacher_w = teacher_weights[w_to_view[0]].flatten().cpu().numpy()
#     student_w = student_weights[w_to_view[0]].flatten().cpu().numpy()

  
#     plt.figure(figsize=(10, 5))
#     plt.hist(teacher_w, bins=100, alpha=0.5, color='blue', label='Teacher', range=(-0.1, 0.1))
#     plt.hist(student_w, bins=100, alpha=0.5, color='red', label='Student', range=(-0.1, 0.1))
#     plt.xlabel('Weight Value')
#     plt.ylabel('Frequency')
#     plt.title('Weight Distributions')
#     plt.legend()
    
    
#     plt.savefig('view_weight.png')
#     plt.show()

# if __name__ == "__main__":
#     main()