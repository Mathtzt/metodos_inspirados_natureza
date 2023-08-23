function [bfit,befit,spbest,spbesth] = SSO(spidn,itern)
%SCO Summary of this function goes here
%   Detailed explanation goes here
    %% Preliminares
    %spidn, spiders number
    %itern, iterations number
    if nargin<1,   spidn=50;     end
    if nargin<2,   itern=500;   end
	%% Function
    f=@Griewank;
    xd=-600;
	xu=600;
    dims = 30;
	for i=1:dims
        lb(i,:)=xd;
        ub(i,:)=xu;
    end
    %% Initial Parameters
    rand('state',0');  % Reset the random generator
    % Define the poblation of females and males
    fpl = 0.65;     % Lower Female Percent
    fpu = 0.9;      % Upper Female Percent
    fp = fpl+(fpu-fpl)*rand;	% Aleatory Percent
    fn = round(spidn*fp);   % Number of females
    mn = spidn-fn;          % Number of males
	%Probabilities of attraction or repulsion
	% Proper tunning for better results
    pm = exp(-(0.1:(3-0.1)/(itern-1):3));
    % Initialization of vectors
    fsp = zeros(fn,dims);   % Initlize females
    msp = zeros(mn,dims);   % Initlize males
    fefit = zeros(fn,1);    % Initlize fitness females
    mafit = zeros(mn,1);    % Initlize fitness males
    spwei = zeros(spidn,1); % Initlize weigth spiders
    fewei = zeros(fn,1); % Initlize weigth spiders
    mawei = zeros(mn,1); % Initlize weigth spiders
    %% Population Initialization
    % Generate Females
    for i=1:fn
        fsp(i,1:dims)=lb(1)+rand(1,dims).*(ub(1)-lb(1));
    end
    % Generate Males
    for i=1:mn
        msp(i,1:dims)=lb(1)+rand(1,dims).*(ub(1)-lb(1));
    end
    %% **** Evaluations *****
	% Evaluation of function for females
    for i=1:fn
        fefit(i)=f(fsp(i,:),dims);
    end
	% Evaluation of function for males
    for i=1:mn
        mafit(i)=f(msp(i,:),dims);
    end
    %% ***** Assign weigth or sort ***********
	% Obtain weight for every spider
    spfit = [fefit' mafit']';   % Mix Females and Males
    bfitw = min(spfit);          % best fitness
    wfit = max(spfit);          % worst fitness
    for i=1:spidn
        spwei(i) = 0.001+((spfit(i)-wfit)/(bfitw-wfit));
    end
    fewei = spwei(1:fn);      % Separate the female mass
    mawei = spwei(fn+1:spidn);% Separate the male mass
    %% Memory of the best
    % Check the best position
    [~,Ibe] = max(spwei);
    % Check if female or male
    if Ibe > fn
        % Is Male
        spbest=msp(Ibe-fn,:);   % Asign best position to spbest
        bfit = mafit(Ibe-fn);      % Get best fitness for memory
    else
        % Is Female
        spbest=fsp(Ibe,:);      % Asign best position to spbest
        bfit = fefit(Ibe);      % Get best fitness for memory
    end
    %% Start the iterations
    for i=1:itern
        %% ***** Movement of spiders *****
        % Move Females
        [fsp] = FeMove(spidn,fn,fsp,msp,spbest,Ibe,spwei,dims,lb,ub,pm(i));
        % Move Males
        [msp] = MaMove(fn,mn,fsp,msp,fewei,mawei,dims,lb,ub,pm(i));
        %% **** Evaluations *****
        % Evaluation of function for females
        for j=1:fn
            fefit(j)=f(fsp(j,:),dims);
        end
        % Evaluation of function for males
        for j=1:mn
            mafit(j)=f(msp(j,:),dims);
        end
        %% ***** Assign weigth or sort ***********
        spfit = [fefit' mafit']';   % Mix Females and Males
        bfitw = min(spfit);          % best fitness
        wfit = max(spfit);          % worst fitness
        % Obtain weight for every spider
        for j=1:spidn
            spwei(j) = 0.001+((spfit(j)-wfit)/(bfitw-wfit));
        end
        fewei = spwei(1:fn);      % Separate the female mass
        mawei = spwei(fn+1:spidn);% Separate the male mass
        %% Mating Operator
        [ofspr] = Mating(fewei,mawei,fsp,msp,dims);
        %% Selection of the Mating
        if isempty(ofspr)
%             % Do nothing
        else
            [fsp,msp,fefit,mafit] = Survive(fsp,msp,ofspr,fefit,mafit,spfit,f,fn,dims);
            % ***** Recalculate the weigth or sort ***********
            spfit = [fefit' mafit']';   % Mix Females and Males
            bfitw = min(spfit);          % best fitness
            wfit = max(spfit);          % worst fitness
            % Obtain weight for every spider
            for j=1:spidn
                spwei(j) = 0.001+((spfit(j)-wfit)/(bfitw-wfit));
            end
            fewei = spwei(1:fn);      % Separate the female mass
            mawei = spwei(fn+1:spidn);% Separate the male mass
        end
        %% Memory of the best
        % Check if best position belongs to male or female
        [~,Ibe2] = max(spwei);
        if Ibe2 > fn
            % Is Male
            spbest2=msp(Ibe2-fn,:);      % Asign best position to spbest
            bfit2 = mafit(Ibe2-fn);      % Get best fitness for memory
        else
            % Is Female
            spbest2 = fsp(Ibe2,:);  % Asign best position to spbest
            bfit2 = fefit(Ibe2);    % Get best fitness for memory
        end
        %% Global Memory
        if bfit<=bfit2
            bfit = bfit;
            spbest = spbest;      % Asign best position to spbest
            befit(i) = bfit;
        else
            bfit = bfit2;
            spbest = spbest2;      % Asign best position to spbest
            befit(i) = bfit;
        end
        spbesth(i,:)=spbest;
%         %% Plot Results
        plot(fsp(:,1),fsp(:,2),'r.',msp(:,1),msp(:,2),'bx',spbest(:,1),spbest(:,2),'go');
        hold on
        axis([lb(1) ub(1) lb(2) ub(2)])
        drawnow
        hold off
    end
    %% Display of results
    figure
    plot(befit)
end