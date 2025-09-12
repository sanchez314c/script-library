############## Open Terminal

############## Clone Ollama Repo
git clone https://github.com/ollama/ollama.git

############## Enter Ollama Dir
cd ollama

############## Edit GPU.go for 35,37 Support
nano /discover/gpu.go

#### Change

var (
	CudaComputeMajorMin = "5"
	CudaComputeMinorMin = "0"
)

#### to

var (
	CudaComputeMajorMin = "3"
	CudaComputeMinorMin = "5"
)

############## Edit CMakePresets.json for 35,37 Support
nano ./CmakePresets.json

#### Change

    {
      "name": "CUDA 11",
      "inherits": [ "CUDA" ],
      "cacheVariables": {
        "CMAKE_CUDA_ARCHITECTURES": "50;52;53;60;61;70;75;80;86"
      }

#### to

    {
      "name": "CUDA 11",
      "inherits": [ "CUDA" ],
      "cacheVariables": {
        "CMAKE_CUDA_ARCHITECTURES": "35;37;50;52;53;60;61;70;75;80;86"
      }
      
############### Compile Source & Build Front-end
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="35;37;50;52" && cmake --build build -- -j18

go generate ./... &&  go build .

############### Serve Ollama
./ollama serve&

############### Run Ollama
./ollama run tinyllama
