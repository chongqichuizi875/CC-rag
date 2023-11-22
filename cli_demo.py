import os
import torch
import platform
import subprocess
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

# model_path = "/mnt/ddata/models/Baichuan2-13B-Base"
model_path = "/mnt/ddata/models/cc-13b-v1-3"
def init_model():
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # device_map="auto",
        load_in_8bit= True,
        trust_remote_code=True
    )
    # .quantize(4).cuda()
    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用百川大模型，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []


def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name).read()
    return text


def main(stream=True):
    model, tokenizer = init_model()
    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
#         prompt = """<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>
# <已知信息>
# 第一节  潍柴动力WP6发动机
# WP6 系列柴油机是潍柴动力与德国道依茨公司合资组建的潍坊潍柴道依茨柴油机有限公司生产制造的、满足国V排放的高速柴油机。该系列柴油机具有结构紧凑，使用可靠，动力性、经济性技术指标优良，起动迅速，操作简单和维修方便等特点，排放指标先进，是货车的理想动力。为使广大用 户更快了解和正确使用、维护 WP6 系列柴油机，特编此维修保养资料。 
# 本章介绍了 WP6 系列车用柴油机的结构特点、操作保养方法、检修要点， 
# 适用于我公司生产的 WP6 系列车用柴油机，供用户参阅。
# 一、发动机使用注意事项
# 1. ECU、共轨油泵和喷油器为精密部件，用户不得拆解。
# 2．增压器转子为高速旋转部件，在机器运转时，严禁任何可移动物体（例如手、工具、棉纱等）接近涡轮增压器的进口，以免对人身或机器造成损害；对转子组件，除涡轮增压器专业维修人员或经潍柴特许的专业维修站点不得拆卸；
# 3．连杆螺栓为一次性使用螺栓，不得重复使用；
# 4．向柴油机添加的机油或燃油，其牌号必须符合使用保养说明书的规定， 并经专用的清洁过滤清器过滤，燃油要经过72小时以上沉淀；在每次开车 前，必须确认冷却液和机油的加入量是否符合要求；
# 5．用户在使用新机时或发动机大修后，应进行50小时试运转，磨合期后需更换机油和机油滤清器滤芯；
# 6．柴油机冷车启动后应慢慢提高转速，不应猛然使其高速运转，也不宜长期 
# 怠速；大负荷运转后，不应立即停车（特殊情况除外），应低速运转5～10 分钟后停车；
# 7．禁止柴油机在无空气滤清器的情况下工作，防止空气未经过滤就进入气 缸。当工作环境恶劣时，应加大空气滤清器滤芯或更换次数，以免造成柴油 机出现早期磨损等故障；
# 8．电气系统各部件的检修必须由电气专业技术人员进行；
# </已知信息>
# <问题>潍柴动力WP6发动机使用注意事项？</问题>"""
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        if prompt.strip() == 'vim':
            prompt = vim_input()
            print(prompt)
        print(Fore.CYAN + Style.BRIGHT + "\nBaichuan 2：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    print(response[position:], end='', flush=True)
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
    # print(torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.current_device(),torch.cuda.get_device_name(0),torch.cuda.get_device_name(1),torch.cuda.get_device_name(2))
    # model = AutoModelForCausalLM.from_pretrained(
    #    model_path,
    #     torch_dtype=torch.float16,
    #     # device_map="auto",
    #     trust_remote_code=True
    # )
    # model = model.quantize(4).cuda()
    # # model.save_pretrained("/mnt/ddata/models/CC-baichuan2-13b-4bit")

    # model.generation_config = GenerationConfig.from_pretrained(
    #     model_path
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path,
    #     use_fast=False,
    #     trust_remote_code=True
    # )
    # tokenizer.save_pretrained("/mnt/ddata/models/CC-baichuan2-13b-4bit")
    # print("save finish")
    # model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto", trust_remote_code=True)
    # model.save_pretrained("/mnt/ddata/models/cc-13b-8bit")
    # tokenizer = AutoTokenizer.from_pretrained(
    #         model_path,
    #         use_fast=False,
    #         trust_remote_code=True
    #     )
    # tokenizer.save_pretrained("/mnt/ddata/models/cc-13b-8bit")
    pass