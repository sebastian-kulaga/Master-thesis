 cd 'E:\MATLAB\R2014a\bin\wrobelwav\blackbird\'
 files = dir('*.wav');
 audio = cell(1,length(files));
 x = 1;
 folder ='E:\MATLAB\R2014a\bin\wrobelwav\out\blackbird10s\';
 bird='blackbird';
 format='.wav';
 for k = 1:length(files)
     audio{k} = wavread(files(k).name);
     [Y,FS,NBITS] = wavread(files(k).name);
     len=length(Y);
     num_samp = round(FS);
     finallength=10*num_samp;
     tmp=floor(len/finallength);
     if tmp==0
         filename=[folder,bird,num2str(x),format];
         wavwrite(Y(y:finallength), FS,NBITS,filename);
         x=x+1;
     else
     for y=1:tmp
           filename=[folder,bird,num2str(x),format];
           wavwrite(Y(((finallength*y-finallength)+1):finallength*y +1), FS,NBITS,filename);
           x=x+1;
     end
     end

 end
