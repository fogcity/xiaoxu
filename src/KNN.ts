import { getShape, range, zeros,abs,sub,argmin,sum, square, most } from "./m"
// 空间复杂度n
export class NearestNeighbor {
    x: any[] =[]
    y: any[] =[]
    constructor(public reg:'L1'|'L2'){

    }
    train(x: any[],y: any[]){
        this.x = x
        this.y = y
    }
 
    predict(x:any,k:number){
        const num_test = getShape(x)[0]
        let Ypred = zeros([num_test])
        for (const i in range(num_test)){
            let d:any[]
            if(this.reg == 'L1'){
                d = this.x.map(v=>sum(abs(sub(v,x[i]))))
            }else{
                d = this.x.map(v=>Math.sqrt(sum(square(sub(v,x[i])))))
            }
            let min_index:number;
            if(k == 1){
                min_index = argmin(d)
            }else{
                const yl = [...d].sort((a,b)=>a-b)
                                 .slice(0,k)
                                 .sort((a,b)=>b-a)          
                const ys =yl.map(v=>this.y[d.indexOf(v)]).flat()
                min_index = most(ys) as number      
            }
            Ypred[i] = this.y[min_index] 
        }
        return Ypred
    }
}

