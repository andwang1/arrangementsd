//
// Created by Luca Grillotti
//

#ifndef VAE_DECODER_HPP
#define VAE_DECODER_HPP

#include <torch/torch.h>

struct DecoderImpl : torch::nn::Module {
    DecoderImpl(int de_hid_dim1, int de_hid_dim2, int de_hid_dim3, int de_hid_dim4, int de_hid_dim5, 
    int de_hid_dim6, int de_hid_dim7,
    int latent_dim) :
        m_tconv_1(torch::nn::ConvTranspose2dOptions(latent_dim, de_hid_dim7, 2)),
        m_tconv_2(torch::nn::ConvTranspose2dOptions(de_hid_dim7, de_hid_dim6, 3)),
        m_tconv_3(torch::nn::ConvTranspose2dOptions(de_hid_dim6, de_hid_dim5, 3)),
        m_tconv_4(torch::nn::ConvTranspose2dOptions(de_hid_dim5, de_hid_dim4, 3)),
        m_tconv_5(torch::nn::ConvTranspose2dOptions(de_hid_dim4, de_hid_dim3, 3)),
        m_tconv_6(torch::nn::ConvTranspose2dOptions(de_hid_dim3, de_hid_dim2, 3)),
        m_tconv_7(torch::nn::ConvTranspose2dOptions(de_hid_dim2, de_hid_dim1, 5)),
        m_tconv_s(torch::nn::ConvTranspose2dOptions(de_hid_dim1, de_hid_dim1, 5).stride(2)),
        m_tconv_m(torch::nn::ConvTranspose2dOptions(de_hid_dim1, 1, 6)),
        m_tconv_v(torch::nn::ConvTranspose2dOptions(de_hid_dim1, 1, 6)),
        m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {
            register_module("m_tconv_1", m_tconv_1);
            register_module("m_tconv_2", m_tconv_2);
            register_module("m_tconv_3", m_tconv_3);
            register_module("m_tconv_4", m_tconv_4);
            register_module("m_tconv_5", m_tconv_5);
            register_module("m_tconv_6", m_tconv_6);
            register_module("m_tconv_7", m_tconv_7);
            register_module("m_tconv_s", m_tconv_s);
            register_module("m_tconv_m", m_tconv_m);
            register_module("m_tconv_v", m_tconv_v);
            _initialise_weights();
        }

        void decode(const torch::Tensor &x, torch::Tensor &mu, torch::Tensor &logvar)
        {
            torch::Tensor out = torch::relu(m_tconv_s(torch::relu(m_tconv_7(torch::relu(m_tconv_6(
                torch::relu(m_tconv_5(torch::relu(m_tconv_4(
                    torch::relu(m_tconv_3(torch::relu(m_tconv_2(torch::relu(m_tconv_1(
                        x.reshape({x.size(0), -1, 1, 1})))))))))))))))));
            mu = m_tconv_m(out).reshape({out.size(0), -1});
            logvar = m_tconv_v(out).reshape({out.size(0), -1});
        }

        void sample_output(const torch::Tensor &mu, const torch::Tensor &logvar, torch::Tensor &output)
        {
            output = torch::randn_like(logvar, torch::device(m_device).requires_grad(true)) * torch::exp(0.5 * logvar) + mu;
        }

        torch::Tensor forward(const torch::Tensor &z, torch::Tensor &logvar) 
        {
            torch::Tensor mu, output;
            decode(z, mu, logvar);
            return mu;
            // if sample output as well
            // sample_output(mu, logvar, output);
            // return output;
        }

        // https://github.com/pytorch/vision/blob/master/torchvision/csrc/models/googlenet.cpp#L150
        void _initialise_weights()
        {
            for (auto& module : modules(/*include_self=*/false)) 
            {
                if (auto M = dynamic_cast<torch::nn::ConvTranspose2dImpl*>(module.get()))
                torch::nn::init::kaiming_normal_(M->weight, 0., torch::kFanIn, torch::kReLU);
            }
        }

        torch::nn::ConvTranspose2d m_tconv_1, m_tconv_2, m_tconv_3, m_tconv_4, m_tconv_5, m_tconv_6, m_tconv_7, 
        m_tconv_s, m_tconv_m, m_tconv_v;
        torch::Device m_device;
};

TORCH_MODULE(Decoder);

#endif //EXAMPLE_PYTORCH_DECODER_HPP
