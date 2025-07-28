

class LLMComiler(nn.Module):
    def __init__(self, config):
        super(Mengzi, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_dir,low_cpu_mem_usage=True)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, context, atten1, query, atten2, answer, atten3):
        # print(answer.shape)
        # print(context.shape)
        out = self.t5(context, atten1, labels=answer)
        print(answer.device)
        print(context.device)
        output = out.logits
        loss = out.loss
        # print(loss)
        
        # last_hidden_states, out = {v for k,v in self.t5(context, atten1, answer, atten3).items()}
        # print(out)
        embedding = self.dropout(output) # [batch_size, max_length, ?]
        # print(embedding.shape)

        return embedding, loss

    def predict_answer(self, context, atten1, query, atten2, answer, atten3):
        embed, loss = self.forward(context, atten1, query, atten2, answer, atten3)
        return embed, loss