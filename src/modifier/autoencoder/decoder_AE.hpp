//
// Created by Luca Grillotti
//

#ifndef AE_DECODER_HPP
#define AE_DECODER_HPP

#include <torch/torch.h>

struct DecoderImpl : torch::nn::Module {
    DecoderImpl(int de_hid_dim1, int de_hid_dim2, int de_hid_dim3, int latent_dim) :
        m_tconv_1(torch::nn::Conv2d(torch::nn::Conv2dOptions(latent_dim, de_hid_dim3, 1))),
        m_tconv_2(torch::nn::Conv2d(torch::nn::Conv2dOptions(de_hid_dim3, de_hid_dim2, 2).transposed(true))),
        m_tconv_s2(torch::nn::Conv2d(torch::nn::Conv2dOptions(de_hid_dim2, de_hid_dim2, 3).stride(2).transposed(true))),
        m_tconv_3(torch::nn::Conv2d(torch::nn::Conv2dOptions(de_hid_dim2, de_hid_dim1, 4).transposed(true))),
        m_tconv_s3(torch::nn::Conv2d(torch::nn::Conv2dOptions(de_hid_dim1, de_hid_dim1, 3).stride(2).transposed(true))),
        m_tconv_4(torch::nn::Conv2d(torch::nn::Conv2dOptions(de_hid_dim1, 1, 4).transposed(true))),
        m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {
            register_module("m_tconv_1", m_tconv_1);
            register_module("m_tconv_2", m_tconv_2);
            register_module("m_tconv_3", m_tconv_3);
            register_module("m_tconv_s2", m_tconv_s2);
            register_module("m_tconv_s3", m_tconv_s3);
            register_module("m_tconv_s4", m_tconv_4);
        }

        torch::Tensor forward(const torch::Tensor &z, torch::Tensor &tmp) 
        {
            return m_tconv_4(torch::relu(m_tconv_s3(torch::relu(m_tconv_3(
                    torch::relu(m_tconv_s2(torch::relu(m_tconv_2(torch::relu(m_tconv_1(
                        z.reshape({z.size(0), 2, 1, 1}))))))))))));
        }

        torch::nn::Conv2d m_tconv_1, m_tconv_2, m_tconv_s2, m_tconv_3, m_tconv_s3, m_tconv_4;
        torch::Device m_device;
};

TORCH_MODULE(Decoder);

#endif //EXAMPLE_PYTORCH_DECODER_HPP
