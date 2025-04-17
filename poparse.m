function [ index8, index7, indexn7] = poparse(trig)
%trig = eegd2.data(:,69);
  

       trigger1 = 204;
       trigger2 = 202;
   
       trigger3 = 100;
       trigger4 = 110;
   

%202-po7 204-po8 nd left-po7
    start_i = [];
   index8 = [];index7 = [];indexn7 = [];

    for i=1:size(trig,1)
        if (trig(i,1) == trigger1) %choose po8

            index8 = [index8 ; i];

        elseif (trig(i,1) == trigger2) %choose po7

            index7 = [index7 ; i];

        elseif (trig(i,1) == trigger3) || (trig(i,1) == trigger4)
            indexn7 = [indexn7 ; i];
        end

    end


   
end