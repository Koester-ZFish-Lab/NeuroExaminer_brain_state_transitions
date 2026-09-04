from pathlib import Path
import numpy as np
import nrrd
import datetime
import subprocess
import platform
import tempfile
import uuid


ants_bin_path = "/Users/koester_lab/install/bin"
ants_use_threads = 8


def convert_path_to_linux(path_name):

    if platform.system() == "Windows":
        path_name_linux = "/mnt/" + str(path_name)

        path_name_linux = path_name_linux.replace("\\", '/')
        path_name_linux = path_name_linux.replace(" ", '\\ ')

        path_name_linux = path_name_linux.replace("C:", 'c')
        path_name_linux = path_name_linux.replace("D:", 'd')
        path_name_linux = path_name_linux.replace("E:", 'e')
        path_name_linux = path_name_linux.replace("F:", 'f')
        path_name_linux = path_name_linux.replace("G:", 'g')
        path_name_linux = path_name_linux.replace("X:", 'x')
        path_name_linux = path_name_linux.replace("Y:", 'y')
        path_name_linux = path_name_linux.replace("Z:", 'z')

        return path_name_linux
    else:
        return str(path_name)


def run_linux_command(command_list, stdin_file=None, stdout_file=None):

    if platform.system() in ["Linux", "Darwin"]:
        # in linux or mac os, we can directly execute the commands
        print(command_list)
        subprocess.run(command_list,
                       stdin=open(stdin_file) if stdin_file is not None else None,
                       stdout=open(stdout_file, "w") if stdout_file is not None else None)

    elif platform.system() == "Windows":
        registration_commands = f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={ants_use_threads}\n" \
                                "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS\n\n"

        registration_commands += ' '.join(command_list)

        if stdin_file is not None:
            registration_commands += f" < {stdin_file}"

        if stdout_file is not None:
            registration_commands += f" > {stdout_file}"

        registration_commands_path = Path(tempfile.gettempdir()) / f"{str(uuid.uuid4())}_linux_commands.sh"
        registration_commands_path_linux = convert_path_to_linux(registration_commands_path)

        commands_file = open(registration_commands_path, 'wb')
        commands_file.write((registration_commands + '\n').encode())
        commands_file.close()

        print("Executing linux script inside windows shell.")
        print(registration_commands)

        subprocess.run(["bash", '-c', registration_commands_path_linux])

        registration_commands_path.unlink()


def compute_volume_registration(source_stack_path, target_stack_path, registration_files_prefix,
                                fixed_mask=None, moving_mask=None):
    print(datetime.datetime.now(), "Running compute_volume_registration.", locals())

    if type(target_stack_path) is not list:
        target_stack_path = [target_stack_path]
    if type(source_stack_path) is not list:
        source_stack_path = [source_stack_path]

    source_stack_path_linux = [source_stack_path[i] for i in range(len(source_stack_path))]
    target_stack_path_linux = [target_stack_path[i] for i in range(len(target_stack_path))]
    registration_files_prefix_linux = registration_files_prefix

    # parameters:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5597853/pdf/gix056.pdf   # most parameters from here
    # https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
    # https://github.com/ANTsX/ANTs/wiki/Tips-for-improving-registration-results

    registration_commands_list = [f"{ants_bin_path}/antsRegistration",
                                  "-v", "1",   # verbose output
                                  "-d", "3",   # dimensionality
                                  "--float", "1",  # float instead of double: single (=lower) precision
                                  "--winsorize-image-intensities", "[0.005, 0.995]",  # for dealing with outliers
                                  "–-use-histogram-matching", "1",  # to adjust input intensities of the images
                                  # so that they match better between our image and the reference brain
                                  # does not seem to work, does not matter if I write 1 or true or anything else
                                  # "--masks", f"[{fixed_mask},{moving_mask}]",  # does not work well
                                  # "--interpolation", "WelchWindowedSinc"  # this is used by almost every paper
                                  # I never used the interpolation keyword and only linear in the apply transform
                                  "-o", f"{registration_files_prefix_linux}_"]  # outputTransformPrefix; format: .nii.gz

    # an initial move of the image is required to bring images roughly in the same space:
    # 0 = use the geometric center of the images
    # 1 = match by center of mass (image intensities)
    # 2 = use the origin of the images
    registration_commands_list += ["--initial-moving-transform"]
    registration_commands_list += [f"[{target_stack_path_linux[0]},{source_stack_path_linux[0]},1]"]

    registration_commands_list += ["-t", "rigid[0.1]"]  # transform; 0.1 is the gradient step
    for i in range(len(target_stack_path)):
        registration_commands_list += ["-m", f"GC[{target_stack_path_linux[i]},{source_stack_path_linux[i]},"
                                             f"1,32,Regular,0.25]"]  # GC for Global Correlation
    registration_commands_list += ["-c", "[200x200x200x0,1e-8,10]",  # convergence
                                   # number of iterations at each level; threshold for the algorithm to stop
                                   "-f", "12x8x4x2",  # shrink factors
                                   # note: always try to register lower resolution to higher resolution!
                                   "-s", "4x3x2x1"]   # smoothing sigmas/values for each step

    registration_commands_list += ["-t", "Affine[0.1]"]
    for i in range(len(target_stack_path)):
        registration_commands_list += ["-m", f"GC[{target_stack_path_linux[i]},{source_stack_path_linux[i]},"
                                             f"1,32,Regular,0.25]"]
    registration_commands_list += ["-c", "[200x200x200x0,1e-8,10]",
                                   "-f", "12x8x4x2",
                                   "-s", "4x3x2x1vox"]  # if no units are specified voxel spacing is automatically used

    registration_commands_list += ["-t", "SyN[0.05,6,0]"]  # gradient step; 0.05 for live, 0.1 for fixed registration
    # updateFieldVarianceInVoxelSpace: default 3, Armin 6; increasing penalty will make image deformation smoother
    # totalFieldVarianceInVoxelSpace: smoothens total deformation, can be 0.5 for live registration
    for i in range(len(target_stack_path)):
        registration_commands_list += ["-m", f"CC[{target_stack_path_linux[i]},{source_stack_path_linux[i]},1,2]"]
        # last number indicates neighbourhood size (pixel radius) used for cross-correlation
        # 2-4 usually works; larger values improve robustness for low-information neighborhoods (reduced contrast)
        # large numbers also increase computation times; Armin 2
    registration_commands_list += ["-c", "[200x200x200x100,1e-7,10]", "-f", "12x8x4x2", "-s", "4x3x2x1"]

    print(registration_commands_list)

    run_linux_command(registration_commands_list)


def apply_volume_registration_to_stack(registration_files_prefix_list, source_stack_path,
                                       target_stack_path, output_stack_path, use_inverted_transforms=False,
                                       interpolation_method="linear", bit='uint16'):
    print(datetime.datetime.now(), "Running apply_volume_registration_to_stack.", locals())

    source_stack_path_linux = source_stack_path
    target_stack_path_linux = target_stack_path
    output_stack_path_linux = output_stack_path

    # get the brightness of the border frame of the source image
    readdata, header = nrrd.read(source_stack_path)
    pad_out = np.percentile(np.r_[readdata[:5].flatten(), readdata[-5:].flatten(), readdata[:, :5].flatten(),
                                  readdata[:, -5:].flatten()], 5)

    registration_commands_list = [f"{ants_bin_path}/antsApplyTransforms",
                                  "--float",    # if nothing is specified, it is 0 by default: double precision used
                                  "-v", "1",    # verbose output
                                  "-d", "3",    # dimensionality
                                  "-f", f"{pad_out}",
                                  # default voxel value to be used when input point maps outside the output domain
                                  "-i", f"{source_stack_path_linux}",
                                  "-r", f"{target_stack_path_linux}",  # reference image (z-brain)
                                  # Linear interpolation is default; most papers use WelchWindowedSinc
                                  "-n", f"{interpolation_method}"]     # used when saving the warped image

    if use_inverted_transforms is False:
        for registration_files_prefix in registration_files_prefix_list:
            registration_files_prefix_linux = registration_files_prefix

            registration_commands_list += ["--transform", f"{registration_files_prefix_linux}_1Warp.nii.gz",
                                           "--transform", f"{registration_files_prefix_linux}_0GenericAffine.mat"]
    else:
        for registration_files_prefix in registration_files_prefix_list[::-1]:
            registration_files_prefix_linux = registration_files_prefix

            registration_commands_list += ["--transform", f"[{registration_files_prefix_linux}_0GenericAffine.mat, 1]",
                                           "--transform", f"{registration_files_prefix_linux}_1InverseWarp.nii.gz"]

    registration_commands_list += ["-o", f"{output_stack_path_linux}"]     # output the warped image
    # usually all outputs are in the .nii.gz file format, but .nrrd is also possible

    run_linux_command(registration_commands_list)

    # ANTS makes it 32bit float, convert it back to 8bit/16bit uint
    readdata, header = nrrd.read(str(output_stack_path))
    readdata = readdata.astype(bit)  # note: use uint8/uint16 depending on template images
    header["encoding"] = 'gzip'  # save a lot of time by not compressing the file: but gets very large then
    header["type"] = bit
    nrrd.write(str(output_stack_path), readdata, header)

    # save time by not writing it to a file but just returning it and directly using it from the memory
    # return readdata, header
