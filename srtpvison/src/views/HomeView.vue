<template>
  <div class="home">
    <el-container>
      <el-header style="text-align:right;font-size: 12px">
        <el-input autosize type="input" :rows="1" placeholder="请输入内容" clearable resize="none" v-model="textarea">
        </el-input>
        <el-button type="primary" icon="el-icon-search" size="mini" @click="returnbyclick(textarea)">搜索</el-button>
      </el-header>
      <el-main>
        <el-table  :data="tableData" v-loading="loading" element-loading-text="检索中,请耐心等待!"
          element-loading-spinner="el-icon-loading" element-loading-background="rgba(0,0,0,0.1)">
          <el-table-column label="序号" width="200" prop="rowid"></el-table-column>
          <el-table-column label="日期" width="200" prop="date"></el-table-column>
          <el-table-column label="相似程度" width="200" prop="similarity"></el-table-column>
          <el-table-column label="执行时间" width="200" prop="executeTime"></el-table-column>
          <el-table-column label="相似代码" width="200">
            <template slot-scope="scope">
              <el-button type="text" @click="openFile(scope.row.rowid)">点击打开相似代码片段</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-main>
      <el-footer>
        <el-upload ref="upload" class="upload-demo" drag accept=".cpp,.c" action="https://t5750t8729.oicp.vip/upload" multiple :auto-upload="false"
          :before-upload="beforeUpload" :on-success="handleSuccess" :on-error="handleError" :limit="1" :on-exceed="handleExceed">
          <i class="el-icon-upload"></i>
          <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
          <div class="el-upload__tip" slot="tip">只能上传c/c++,且不超过500kb,且一次只能上传一个文件</div>
        </el-upload>
        <el-alert ref="myalert"
        v-if="isalert"
        :title="alerttitle"
        type="error"
        show-icon
        center
        closable
        :close="handleclose"
        >
      </el-alert>
        <el-button style="margin-left: 10px;" size="small" type="success" @click="submitUpload">上传到服务器</el-button>
      </el-footer>
    </el-container>
  </div>
</template>
<style lang="less" scoped>
.el-header {
  background-color: #B3C0D1;
  color: #333;
  line-height: 60px;
  display: flex;
}

.el-input {
  flex: 1;
}

.el-button {
 
  border: none;
  height: 50%;
  position: relative;
  margin: 15px;
  z-index: 100;

}
</style>
<script>
import axios from 'axios'
export default {
  name: 'HomeView',
  data() {
    return {
      tableData: [],
      textarea: '',
      rowid: 1,
      loading: false,
      isalert: false,
      alerttitle: ''
    }
  },
  methods: {
    handleclose(){
      this.alerttitle = ''
    },
    handleExceed(files,fileList){
      this.isalert = true;
      this.alerttitle = "你选择的文件过多"
    },
    submitUpload(){
      this.$refs.upload.submit();
    },
    openFile(ID) {
      this.$alert("<div style='height:200px;overflow:auto;'>" + this.tableData[ID].similarCode.replace(/\n/g, '<br>') + "</div>", '相似代码片段', {
        confirmButtonText: '确定',
        dangerouslyUseHTMLString: true,
        lockScroll: false
      });
    },
    saveToFile(text) {
      const blob = new Blob([text], { type: 'text/plain' })
      blob.lastModifiedDate = new Date()
      blob.name = 'my-file.cpp'
      return blob
    },
    returnbyclick(textToSearch) {
      this.loading = true
      const formData = new FormData()
      formData.append('file', this.saveToFile(textToSearch))

      return axios.post('https://t5750t8729.oicp.vip/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }).then(response => {
        this.handleresponse(response.data)
      })
    },
    handlesort(tableData) {
        let rowlist = []
        for (let val in tableData){
            console.log(val)
            rowlist.push(val)
        }
        console.log(rowlist)
        rowlist.sort(function(a,b) {return a-b});
        for (let i = 0;i < tableData.length;i++){
            tableData[i].rowid = rowlist[i]
        }
        console.log(rowlist)
        console.log(this.tableData)
    },
    beforeUpload(file) {
      this.loading = true;
    },
    handleresponse(response) {
      console.log(response)
      // 处理成功的回调
      let returnCode = response[2]     //返回的代码片段
      let returnExtent = response[3]    //返回的相似度，high or low
      let returnCodeList = []
      let returnExtentList = []
      let lengthArray = 0   // 统计返回的代码条数
      for (let val in returnCode) {
        returnCodeList.push(returnCode[val])
        lengthArray += 1
      }
      for (let val in returnExtent) {
        returnExtentList.push(returnExtent[val])
      }
      for (let i = 0; i < lengthArray; i++) {
        let item = { "rowid": this.rowid++, "date": response[0], "executeTime": response[1], "similarCode": returnCodeList[i], "similarity": returnExtentList[i] }
        if (returnExtentList[i] == 'high') {
          this.tableData.unshift(item)
        }
        else {
          this.tableData.push(item)
        }
        
      }
      this.handlesort(this.tableData)
      this.loading = false

    },
    handleSuccess(response, file) {
      this.handleresponse(response)
      this.loading = false
    },
    handleError(error, file) {
      // 处理失败的回调
      this.loading = false
      alert("服务器开小差了？")
    },
  }
}
</script>
