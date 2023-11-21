from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs import (LLM_MODELS, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.utils import get_doc_path
import json
import math
import os
from pathlib import Path
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from langchain import PromptTemplate


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                            knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                            top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                            score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=2),
                            history: List[History] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                            stream: bool = Body(False, description="流式输出"),
                            model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                            temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                            max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                            prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                            request: Request = None,
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]
    
    
    #用户意图
    _model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
            )
    # intention_prompt_template = get_prompt_template("knowledge_base_chat", "intention")
    # intention_input_msg = History(role="system", content=intention_prompt_template).to_msg_template(False)
    # intention_chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in []] + [intention_input_msg])
    # 你是军事装备维修领域专家。
    intention_chat_prompt =PromptTemplate(
        template="请根据用户输入判断是否是咨询军事装备的专业问题，如果是请返回'是'，否则返回'否'。\
        答案格式要求: '是'或者'否'。不能有其他答案 \
        参考示例: \
            示例1: \
                用户输入: \
                    '你叫什么名字' \
                返回: \
                    '否' \
            示例2: \
                用户输入: \
                    '今天天气怎么样？'\
                        \
                返回: \
                    '否' \
            示例3: \
                用户输入: \
                    '能介绍一下杭州这个城市？'\
                返回: \
                    '否' \
            示例4: \
                用户输入: \
                    '车辆发动机故障怎么维修？'\
                返回: \
                    '是' \
            示例5: \
                用户输入: \
                    '陆军装备有哪些？' \
                返回: \
                    '是' \
        用户输入: \
            '{question}' \
        返回: \
        ",input_variables=["question"]
    )
    intention_chain = LLMChain(llm=_model,prompt=intention_chat_prompt)
    intention = intention_chain({"question":query})
    print("xxxx","用户意图",intention)
    chat_intention  = "否" == intention.get("text","是")
    
    

    async def knowledge_base_chat_iterator(query: str,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODELS[0],
                                           prompt_name: str = prompt_name,
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        if chat_intention:
            docs = []
            context =""
        else:
            docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
            context = "\n".join([doc.page_content for doc in docs])
        
        if len(docs) == 0: ## 如果没有找到相关文档，使用Empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "Empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)
        

        # txt_block_len = 1024
        # if len(context)>txt_block_len*2:
        #     # print(">>>>","context",context)
        #     ctemp = []
        #     summary_prompt_template = get_prompt_template("knowledge_base_chat", "abstract")
        #     _input_msg = History(role="user", content=summary_prompt_template).to_msg_template(False)
        #     _chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in []] + [_input_msg])
            
        #     _model = get_ChatOpenAI(
        #         model_name=model_name,
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #         )
        #     _summary_chain = LLMChain(prompt=_chat_prompt, llm=_model)
            
        #     for txt_block_index in range(math.ceil(len(context)/txt_block_len)):
        #         start= txt_block_index*txt_block_len
        #         end = min((txt_block_index+1)*txt_block_len,len(context))
        #         raw = context[start:end]            
        #         # summary_query = summary_prompt_template.format(raw)
        #         # print(">>>>","summary_query",txt_block_index,start,end,raw)
        #         summary = _summary_chain({"context":raw})
                
        #         # print(">>>>","summary",summary['text'])
        #         ctemp.append(summary['text'])
            
        #     context  = "".join(ctemp)
        #     print(">>>>","总结后的内容",context)
        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        ref_documents=[]
        doc_path = get_doc_path(knowledge_base_name)
        print(f"docs: {docs}")
        for inum, doc in enumerate(docs):
            filename = Path(doc.metadata["source"]).resolve().relative_to(doc_path)
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n{doc.score}\n\n"""

            source_documents.append(text)
            ref_documents.append({"filename":str(os.path.basename(doc.metadata["source"])),"url":str(url),"score":doc.score,"content_pos_1":{"page_no":1,"x":0,"y":0.5},"content_pos_2":{"page_no":1,"x":1,"y":0.8}})

        if len(source_documents) == 0: # 没有找到相关文档
            source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            # yield json.dumps({"docs": source_documents}, ensure_ascii=False)

            s = json.dumps({"ref_docs": ref_documents}, ensure_ascii=False)
            yield s
            
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return StreamingResponse(knowledge_base_chat_iterator(query=query,
                                                          top_k=top_k,
                                                          history=history,
                                                          model_name=model_name,
                                                          prompt_name=prompt_name),
                             media_type="text/event-stream")