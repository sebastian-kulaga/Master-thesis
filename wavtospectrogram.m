 cd 'E:\MATLAB\R2014a\bin\wrobelwav\out\blackbird10s\'
 files = dir('*.wav');
 audio = cell(1,length(files));
%  bird='raven';
 format='.png';
 
 folderS ='E:\MATLAB\R2014a\bin\wrobelwav\out\specblackbird10\';
 len=length(files);
 for k = 1:len
%      audio{k} = wavread(files(k).name);
     [Y,FS,NBITS] = wavread(files(k).name);
     newname=files(k).name;
     newname=newname(1:end-4);
     filename=[folderS,newname,format];
     myNewFig = figure('renderer','zbuffer','visible','off');
     spectrogram(Y, hamming(2048), [], [], FS,'yaxis');
     axis off
     set(gca,'position',[0 0 1 1],'units','normalized')
     saveas(gcf,filename);
     clear files(k);
 end