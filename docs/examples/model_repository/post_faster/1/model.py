# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sys
import json
from PIL import Image
import cv2
from torchvision import transforms
import torch

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_POST_FASTER")
        
        self.output_shape = np.array(output0_config['dims']) #[3 244 244]

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])


    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        output0_dtype = self.output0_dtype
        responses = []
        tranform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(),                                         
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

        for request in requests:
            imgs = pb_utils.get_input_tensor_by_name(request, "INPUT_POST_FASTER_IMG").as_numpy()
            dets = pb_utils.get_input_tensor_by_name(request, "INPUT_POST_FASTER").as_numpy()
            # print("*****************************************")
            # print(imgs.shape)
            # print("-----------------------------------------")
            # print(img)
            
            #img=np.transpose(img, (1,2,0)).astype(np.uint8)
            #imgs=np.transpose(imgs, (0,2,3,1)).astype(np.uint8)
            
            
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            # print(imgs.shape)
            # print(img.shape)
            # print("---------------------------")
            # print(dets)
            # print(dets.shape)
            # print(type(img))
            
            # print("---------------------------")
            # print(dets)
            # print(dets.shape)
            # print(">>>>>>>>>>>>")
            # print(dets[dets[:,4]>=0.5])
            
            
            #dets=dets[dets[:,4]>=0.5]
            '''dets=dets[:, :, dets[:,:,4]>=0.5]
            
            
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(dets.shape)'''
            #------------------------------------mac dinh chi co duy nhat mot vat the-------------------------------------------
            #x,y,w,h=300, 350, 224, 224#dets[0][0],dets[0][1],dets[0][2],dets[0][3]
            #imgs=imgs[:, :, int(x-w/2):int(x+w/2), int(y-h/2):int(y+h/2)]
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print(imgs.shape)
            # print(imgs)
            x,y,w,h=dets[0][0][0],dets[0][0][1],dets[0][0][2],dets[0][0][3]
            # print("********************")
            # print(x,y,w,h)
            img0=np.array(np.transpose(imgs[0], (1,2,0)))
            # print("::::::::::::::::::::::::::::::::")
            # print(img0)
            cropped_imgs=img0[int(x):int(w), int(y):int(h)]
            # print("####################################")
            # print(cropped_imgs.shape)
            
            # print("&&&&&&&&&&&&&&&&$$$$$$$$$$$$")
            # print(cropped_imgs.shape)
            cropped_imgs=cv2.resize(cropped_imgs, (self.output_shape[1], self.output_shape[2]))
            cropped_imgs=np.transpose(cropped_imgs, (2,0,1))
            # print(">>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>")
            # print(cropped_imgs.shape)
            cropped_imgs=np.expand_dims(cropped_imgs,0)
            imgs=cropped_imgs
            
            # print("+++++++++++++++++++++++++++++++++++++=8")
            # print(cropped_imgs.shape)
            
            #img_rgb= Image.fromarray((img * 1).astype(np.uint8))
            #.convert('RGB')
            #print("+++++++++++++++++++++++++++++++++++++=8")
            #print(img_rgb.shape)
            #imgs= tranform(cropped_imgs)
            # print("^^^^^^^^^^^^^^^^^")
            # print(img)
            # print(img.shape)'''
            #imgs=imgs[:, :, int(x-w/2):int(x+w/2), int(y-h/2):int(y+h/2)]
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            # print(imgs.shape)
            
             
            

            out_tensor_0 = pb_utils.Tensor("OUTPUT_POST_FASTER", imgs.astype(output0_dtype))
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        
        return responses


def finalize(self):
    """`finalize` is called only once when the model is being unloaded.
    Implementing `finalize` function is OPTIONAL. This function allows
    the model to perform any necessary clean ups before exit.
    """
    print('Preprocessed image !!!')
